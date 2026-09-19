"""Quantized dosage store: uint8 ALT-dosage codes in variant-major Zarr v3 shards.

Layout ("svpgs-store v1", target-format REPORT §2 with addenda A1/A2)::

    MANIFEST.json                        format, chromosomes, records per chromosome, samples per
                                         half, md5 of each chromosome's (pos, ref_len, alt_len)
    dosage/half{h}/{chrom}/              Zarr v3 uint8 [records, samples_h], sharded by record rows
    variants/{chrom}/{column}/           Zarr v3 1-D columns, one row per record
    stats/half{h}/{chrom}/{column}/      exact integer per-record sums over the half's samples

Encoding: ``code = (DS_milli * 127 + 500) // 1000`` in [0, 254] and DS = code / 127, so 0, 1
and 2 are exact and the error is at most 1/254.  Code 255 is the Zarr fill value and is never
written.  Dosages count the ALT allele.  The Stage 0 kernels (``genotype_buffers``) work on
``s = code - 127`` in [-127, 127]; standardization is affine in s, so (DS - mean_DS) / sd_DS =
(s - mean_s) / sd_s exactly.

Two inner-chunk codec chains exist, and each array's ``zarr.json`` says which one it uses:

- ``[bytes, zstd(3), crc32c]``, the bucket store.  Each 64-record chunk is an independent zstd
  frame whose crc32c is verified whenever it is decoded.
- ``[bytes]``, the local NVMe/RAM cache.  A shard's records are one contiguous byte range, so a
  range inside a shard is a zero-copy view of the page cache.  Any range can also be read with
  ``preadv`` straight into a caller buffer, such as pinned host memory for GPU staging.

The shard index is crc32c-checked in both chains.  Variant metadata is columnar; no Python
object is built per variant.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import resource
import threading
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import google_crc32c
import numpy as np
import zstandard

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array
from sv_pgs.compute_budget import ComputeBudget, _try_import_cupy
from sv_pgs.config import VariantClass


class _PinnedBufferPool:
    """Process-wide pinned host buffer pool.

    Pinning host memory via ``cudaHostAlloc`` (what CuPy's
    ``alloc_pinned_memory`` wraps) is expensive: each call requires the
    kernel to lock pages and update the IOMMU, which for a 7+ GB
    bitpacked-cache staging buffer can cost a meaningful fraction of a
    minute. Freeing the buffer unmaps it; the next call immediately
    reallocates and re-pins from scratch. When the pipeline runs SNP-only
    then SNP+SV in the same process, or iterates the disease loop with
    bitpacked cache loads at the head of each disease, the 7 GB pin/unpin
    churn becomes a real wall-time tax.

    This pool keeps released allocations around (keyed by size) so the
    next ``acquire(n)`` of a same-or-smaller request reuses an existing
    pin instead of round-tripping through the kernel. Grows monotonically
    — we never shrink — and is bounded only by the host-RAM budget the
    caller already enforces upstream.

    Thread-safe under a module-level ``threading.Lock``. The lock is
    released across the (potentially multi-second) actual
    ``alloc_pinned_memory`` call so concurrent acquires of pool-hit sizes
    are not serialized behind a cold-allocate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._available: list[tuple[int, Any]] = []
        self._in_flight: dict[int, tuple[int, Any]] = {}
        self._n_allocs = 0
        self._n_reuses = 0
        self._peak_total_bytes = 0

    def acquire(self, cp: Any, nbytes: int) -> tuple[Any, np.ndarray]:
        """Return ``(pinned_mem, uint8 numpy view of length ``nbytes``)``.

        Best-fit search: smallest available buffer ≥ ``nbytes``. Falls
        back to a fresh ``alloc_pinned_memory`` if no candidate fits.
        """
        if nbytes <= 0:
            return None, np.empty((0,), dtype=np.uint8)
        nbytes = int(nbytes)
        with self._lock:
            best_idx = -1
            best_size = -1
            for idx, (sz, _mem) in enumerate(self._available):
                if sz >= nbytes and (best_idx < 0 or sz < best_size):
                    best_idx = idx
                    best_size = sz
            if best_idx >= 0:
                sz, mem = self._available.pop(best_idx)
                self._in_flight[id(mem)] = (sz, mem)
                self._n_reuses += 1
                view = np.frombuffer(mem, dtype=np.uint8, count=nbytes)
                return mem, view
        # Allocate outside the lock — pinning a multi-GB region can take
        # seconds and we don't want every other thread blocked on it.
        pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
        with self._lock:
            self._in_flight[id(pinned_mem)] = (nbytes, pinned_mem)
            self._n_allocs += 1
            total = sum(sz for sz, _ in self._available) + sum(
                sz for sz, _ in self._in_flight.values()
            )
            if total > self._peak_total_bytes:
                self._peak_total_bytes = total
        view = np.frombuffer(pinned_mem, dtype=np.uint8, count=nbytes)
        return pinned_mem, view

    def release(self, mem: Any) -> None:
        """Return ``mem`` to the pool so a later acquire can reuse it.

        Safe with ``None`` (no-op) and on double-release (drops silently).
        """
        if mem is None:
            return
        with self._lock:
            entry = self._in_flight.pop(id(mem), None)
            if entry is None:
                return
            self._available.append(entry)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "available_count": len(self._available),
                "available_bytes": sum(sz for sz, _ in self._available),
                "in_flight_count": len(self._in_flight),
                "in_flight_bytes": sum(sz for sz, _ in self._in_flight.values()),
                "allocs": self._n_allocs,
                "reuses": self._n_reuses,
                "peak_total_bytes": self._peak_total_bytes,
            }


_PINNED_POOL = _PinnedBufferPool()


def _pinned_pool() -> _PinnedBufferPool:
    """Return the process-wide pinned-buffer pool."""
    return _PINNED_POOL


def _allocate_pinned(cp: Any, nbytes: int) -> tuple[Any, np.ndarray]:
    """Acquire a pinned-host uint8 staging buffer from the process-wide pool.

    Returns ``(pinned_mem, numpy_view)``. Pass ``pinned_mem`` to
    ``_release_pinned`` when the buffer is no longer needed so a later
    acquire can reuse it instead of re-pinning multi-GB regions from
    scratch.
    """
    return _PINNED_POOL.acquire(cp, int(nbytes))


def _release_pinned(mem: Any) -> None:
    """Return a pinned buffer to the pool. Safe with ``None`` and on double-release."""
    _PINNED_POOL.release(mem)


STORE_FORMAT = "svpgs-store-v1"
MANIFEST_FILE = "MANIFEST.json"
CODES_PER_DOSAGE = 127
MAXIMUM_CODE = 254
MISSING_CODE = 255
MAXIMUM_DOSAGE_MILLI = 2000
DEFAULT_SHARD_ROWS = 65536
DEFAULT_INNER_CHUNK_ROWS = 64
ZSTD_LEVEL = 3
VARIANT_CLASSES = tuple(VariantClass)
# On-disk variant columns every store carries; any other column is a prior annotation.
REQUIRED_VARIANT_COLUMNS = ("pos", "ref_len", "alt_len", "cm", "variant_class", "group_first")
Codec = Literal["raw", "zstd"]

_ZARR_METADATA_FILE = "zarr.json"
_UNWRITTEN_CHUNK = np.uint64(2**64 - 1)
_SHARD_INDEX_ENTRY_BYTES = 16
_CRC32C_BYTES = 4
_ID_BYTES_COLUMN = "id_bytes"
_ID_OFFSETS_COLUMN = "id_offsets"
_ZARR_DATA_TYPES = {
    np.dtype(np.bool_): "bool",
    np.dtype(np.int8): "int8",
    np.dtype(np.int16): "int16",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.uint64): "uint64",
    np.dtype(np.float16): "float16",
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
}
_NUMPY_DATA_TYPES = {name: dtype for dtype, name in _ZARR_DATA_TYPES.items()}
_BYTES_CODEC = {"name": "bytes"}
_LITTLE_ENDIAN_BYTES_CODEC = {"name": "bytes", "configuration": {"endian": "little"}}
_CRC32C_CODEC = {"name": "crc32c"}
_INNER_CODECS: dict[str, list[dict[str, Any]]] = {
    "raw": [_BYTES_CODEC],
    "zstd": [_BYTES_CODEC, {"name": "zstd", "configuration": {"level": ZSTD_LEVEL, "checksum": False}}, _CRC32C_CODEC],
}


def encode_dosage_milli(dosage_milli: NDArray) -> U8Array:
    """Map integer thousandths of ALT dosage (0..2000) to stored codes (0..254), rounding half up."""
    milli = np.asarray(dosage_milli)
    if milli.dtype.kind not in "iu":
        raise TypeError(f"dosage_milli must be an integer array; got dtype {milli.dtype}.")
    if milli.size and (int(milli.min()) < 0 or int(milli.max()) > MAXIMUM_DOSAGE_MILLI):
        raise ValueError(f"dosage_milli must lie in [0, {MAXIMUM_DOSAGE_MILLI}].")
    return ((milli.astype(np.uint32) * CODES_PER_DOSAGE + 500) // 1000).astype(np.uint8)


def _reject_missing_codes(codes: U8Array) -> None:
    if codes.size and int(codes.max()) > MAXIMUM_CODE:
        raise ValueError(f"code {MISSING_CODE} is the missing-dosage fill value and is never stored.")


# ---------------------------------------------------------------------------
# Zarr v3 containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CodeArrayLayout:
    """Geometry and inner-chunk codec of one sharded uint8 code array [row_count, sample_count]."""

    row_count: int
    sample_count: int
    shard_rows: int
    inner_rows: int
    codec: Codec

    def __post_init__(self) -> None:
        if self.row_count < 1 or self.sample_count < 1:
            raise ValueError("a code array needs at least one row and one sample.")
        if self.inner_rows < 1 or self.shard_rows % self.inner_rows != 0:
            raise ValueError("shard_rows must be a positive multiple of inner_rows.")
        if self.codec not in _INNER_CODECS:
            raise ValueError(f"codec must be one of {sorted(_INNER_CODECS)}.")

    @property
    def shard_count(self) -> int:
        return -(-self.row_count // self.shard_rows)

    @property
    def inner_chunks_per_shard(self) -> int:
        return self.shard_rows // self.inner_rows

    @property
    def inner_chunk_bytes(self) -> int:
        return self.inner_rows * self.sample_count

    @property
    def shard_index_bytes(self) -> int:
        return self.inner_chunks_per_shard * _SHARD_INDEX_ENTRY_BYTES + _CRC32C_BYTES

    def shard_row_count(self, shard_index: int) -> int:
        return min(self.shard_rows, self.row_count - shard_index * self.shard_rows)

    def shard_written_chunks(self, shard_index: int) -> int:
        return -(-self.shard_row_count(shard_index) // self.inner_rows)


def _code_array_metadata(layout: CodeArrayLayout) -> dict[str, Any]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [layout.row_count, layout.sample_count],
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [layout.shard_rows, layout.sample_count]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": MISSING_CODE,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [layout.inner_rows, layout.sample_count],
                    "codecs": _INNER_CODECS[layout.codec],
                    "index_codecs": [_LITTLE_ENDIAN_BYTES_CODEC, _CRC32C_CODEC],
                    "index_location": "end",
                },
            }
        ],
        "attributes": {"format": STORE_FORMAT, "encoding": "code = (ds_milli * 127 + 500) // 1000"},
        "dimension_names": ["variant", "sample"],
    }


def _require(condition: bool, directory: Path, detail: str) -> None:
    if not condition:
        raise ValueError(f"unsupported Zarr array at {directory}: {detail}")


def _read_metadata(directory: Path) -> dict[str, Any]:
    return json.loads((directory / _ZARR_METADATA_FILE).read_text())


def _layout_from_metadata(metadata: Mapping[str, Any], directory: Path) -> CodeArrayLayout:
    _require(metadata.get("zarr_format") == 3 and metadata.get("node_type") == "array", directory, "not a Zarr v3 array")
    _require(metadata.get("data_type") == "uint8", directory, "data_type must be uint8")
    shape = metadata["shape"]
    _require(len(shape) == 2, directory, "shape must be [records, samples]")
    grid = metadata["chunk_grid"]
    _require(grid["name"] == "regular", directory, "chunk_grid must be regular")
    shard_shape = grid["configuration"]["chunk_shape"]
    _require(shard_shape[1] == shape[1], directory, "a shard must span every sample")
    key_encoding = metadata["chunk_key_encoding"]
    _require(
        key_encoding["name"] == "default" and key_encoding["configuration"]["separator"] == "/",
        directory,
        "chunk keys must use the default encoding with '/'",
    )
    codecs = metadata["codecs"]
    _require(len(codecs) == 1 and codecs[0]["name"] == "sharding_indexed", directory, "codecs must be one sharding_indexed")
    sharding = codecs[0]["configuration"]
    inner_shape = sharding["chunk_shape"]
    _require(inner_shape[1] == shape[1], directory, "an inner chunk must span every sample")
    matching = [name for name, chain in _INNER_CODECS.items() if sharding["codecs"] == chain]
    _require(len(matching) == 1, directory, f"inner codecs must be one of {list(_INNER_CODECS.values())}")
    _require(
        sharding["index_codecs"] == [_LITTLE_ENDIAN_BYTES_CODEC, _CRC32C_CODEC],
        directory,
        "shard index must be little-endian bytes + crc32c",
    )
    _require(sharding["index_location"] == "end", directory, "shard index must sit at the end of the shard")
    return CodeArrayLayout(
        row_count=int(shape[0]),
        sample_count=int(shape[1]),
        shard_rows=int(shard_shape[0]),
        inner_rows=int(inner_shape[0]),
        codec="raw" if matching[0] == "raw" else "zstd",
    )


def create_code_array(
    directory: Path,
    row_count: int,
    sample_count: int,
    *,
    codec: Codec,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    inner_rows: int = DEFAULT_INNER_CHUNK_ROWS,
) -> CodeArrayLayout:
    """Write the Zarr v3 metadata of an empty code array and return its layout."""
    layout = CodeArrayLayout(
        row_count=row_count, sample_count=sample_count, shard_rows=shard_rows, inner_rows=inner_rows, codec=codec
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _ZARR_METADATA_FILE).write_text(json.dumps(_code_array_metadata(layout), indent=1))
    return layout


def _shard_path(directory: Path, shard_index: int) -> Path:
    return directory / "c" / str(shard_index) / "0"


_COMPRESSION_POOL: list[ThreadPoolExecutor] = []
_COMPRESSION_POOL_LOCK = threading.Lock()
# A forked child inherits the pool object but not its threads; it builds its own on first use.
os.register_at_fork(after_in_child=_COMPRESSION_POOL.clear)


def _compression_pool() -> ThreadPoolExecutor:
    """The process's zstd worker threads (zstd releases the GIL), one per usable CPU."""
    with _COMPRESSION_POOL_LOCK:
        if not _COMPRESSION_POOL:
            _COMPRESSION_POOL.append(
                ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0)), thread_name_prefix="dosage-store-zstd")
            )
        return _COMPRESSION_POOL[0]


def _compressed_chunk(payload: bytes) -> bytes:
    frame = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(payload)
    return frame + google_crc32c.value(frame).to_bytes(_CRC32C_BYTES, "little")


class CodeShardWriter:
    """Stream the rows of one shard of a code array into its shard file.

    Rows arrive in order and are cut into inner chunks; the last one is padded with the fill
    value.  zstd chunks are compressed on the process's worker threads, a bounded number in
    flight, and written in order.  The file is written under a temporary name and renamed into
    place by ``close`` only after every chunk and the checksummed index are on disk.  Shards are
    independent files, so separate processes may write separate shards of one array concurrently.
    """

    def __init__(self, directory: Path, layout: CodeArrayLayout, shard_index: int) -> None:
        if not 0 <= shard_index < layout.shard_count:
            raise ValueError(f"shard {shard_index} is outside the array's {layout.shard_count} shards.")
        self._layout = layout
        self._shard_index = shard_index
        self._final_path = _shard_path(directory, shard_index)
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        self._partial_path = self._final_path.with_name(self._final_path.name + ".partial")
        self._handle = open(self._partial_path, "wb")
        self._rows_expected = layout.shard_row_count(shard_index)
        self._rows_written = 0
        self._chunk = np.full((layout.inner_rows, layout.sample_count), MISSING_CODE, dtype=np.uint8)
        self._chunk_rows = 0
        self._index = np.full((layout.inner_chunks_per_shard, 2), _UNWRITTEN_CHUNK, dtype="<u8")
        self._chunks_written = 0
        self._bytes_written = 0
        self._compressing: deque[Future[bytes]] = deque()
        self._compressing_limit = 2 * len(os.sched_getaffinity(0))

    def _write_payload(self, payload: bytes) -> None:
        self._handle.write(payload)
        self._index[self._chunks_written] = (self._bytes_written, len(payload))
        self._bytes_written += len(payload)
        self._chunks_written += 1

    def _emit_chunk(self) -> None:
        payload = self._chunk.tobytes()
        if self._layout.codec == "raw":
            self._write_payload(payload)
        else:
            self._compressing.append(_compression_pool().submit(_compressed_chunk, payload))
            while len(self._compressing) > self._compressing_limit:
                self._write_payload(self._compressing.popleft().result())
        self._chunk.fill(MISSING_CODE)
        self._chunk_rows = 0

    def write_rows(self, codes: U8Array) -> None:
        if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != self._layout.sample_count:
            raise ValueError(f"rows must be uint8 [rows, {self._layout.sample_count}]; got {codes.dtype} {codes.shape}.")
        if self._rows_written + codes.shape[0] > self._rows_expected:
            raise ValueError(f"shard {self._shard_index} holds {self._rows_expected} rows; too many written.")
        _reject_missing_codes(codes)
        cursor = 0
        while cursor < codes.shape[0]:
            take = min(self._layout.inner_rows - self._chunk_rows, codes.shape[0] - cursor)
            self._chunk[self._chunk_rows : self._chunk_rows + take] = codes[cursor : cursor + take]
            self._chunk_rows += take
            cursor += take
            if self._chunk_rows == self._layout.inner_rows:
                self._emit_chunk()
        self._rows_written += codes.shape[0]

    def close(self) -> None:
        if self._rows_written != self._rows_expected:
            raise ValueError(
                f"shard {self._shard_index} received {self._rows_written} of its {self._rows_expected} rows."
            )
        if self._chunk_rows:
            self._emit_chunk()
        while self._compressing:
            self._write_payload(self._compressing.popleft().result())
        index_bytes = self._index.tobytes()
        self._handle.write(index_bytes)
        self._handle.write(google_crc32c.value(index_bytes).to_bytes(_CRC32C_BYTES, "little"))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.replace(self._partial_path, self._final_path)

    def __enter__(self) -> CodeShardWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
            return
        self._handle.close()
        self._partial_path.unlink()


@dataclass(slots=True)
class _OpenShard:
    descriptor: int
    chunk_offsets: NDArray
    chunk_sizes: NDArray
    data_bytes: int
    mapping: mmap.mmap | None = None


def _pread_exact(descriptor: int, buffers: list[memoryview], offset: int) -> None:
    """Fill every buffer from ``descriptor`` starting at ``offset``; short reads are resumed."""
    vector_limit = int(os.sysconf("SC_IOV_MAX"))
    pending = deque(buffer for buffer in buffers if len(buffer))
    while pending:
        batch = [pending[position] for position in range(min(vector_limit, len(pending)))]
        read_bytes = os.preadv(descriptor, batch, offset)
        if read_bytes == 0:
            raise EOFError(f"shard file ended {sum(len(buffer) for buffer in pending)} bytes early.")
        offset += read_bytes
        while read_bytes:
            head = pending[0]
            if read_bytes >= len(head):
                read_bytes -= len(head)
                pending.popleft()
            else:
                pending[0] = head[read_bytes:]
                read_bytes = 0


def _row_buffers(target: U8Array) -> list[memoryview]:
    return [memoryview(target.reshape(-1))] if target.flags.c_contiguous else [memoryview(row) for row in target]


_THREAD_STATE = threading.local()


def _decompressor() -> zstandard.ZstdDecompressor:
    if not hasattr(_THREAD_STATE, "decompressor"):
        _THREAD_STATE.decompressor = zstandard.ZstdDecompressor()
    return _THREAD_STATE.decompressor


def _scratch(name: str, byte_count: int) -> U8Array:
    """A per-thread reusable byte buffer, so reads do not fault in fresh pages on every call."""
    buffer = getattr(_THREAD_STATE, name, None)
    if buffer is None or buffer.size < byte_count:
        buffer = np.empty(byte_count, dtype=np.uint8)
        setattr(_THREAD_STATE, name, buffer)
    return buffer[:byte_count]


class CodeArray:
    """Read access to one sharded uint8 code array written by :class:`CodeShardWriter`."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.layout = _layout_from_metadata(_read_metadata(self.directory), self.directory)
        self._open_shards: dict[int, _OpenShard] = {}
        self._lock = threading.Lock()

    def _open_shard(self, shard_index: int) -> _OpenShard:
        layout = self.layout
        path = _shard_path(self.directory, shard_index)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            file_bytes = os.fstat(descriptor).st_size
            data_bytes = file_bytes - layout.shard_index_bytes
            index_bytes = os.pread(descriptor, layout.shard_index_bytes, data_bytes)
            index_body = index_bytes[:-_CRC32C_BYTES]
            if data_bytes < 0 or google_crc32c.value(index_body) != int.from_bytes(index_bytes[-_CRC32C_BYTES:], "little"):
                raise ValueError(f"{path}: shard index fails its crc32c check.")
            index = np.frombuffer(index_body, dtype="<u8").reshape(layout.inner_chunks_per_shard, 2)
            written = layout.shard_written_chunks(shard_index)
            offsets, sizes = index[:written, 0].astype(np.int64), index[:written, 1].astype(np.int64)
            if np.any(index[:written] == _UNWRITTEN_CHUNK) or not np.all(index[written:] == _UNWRITTEN_CHUNK):
                raise ValueError(f"{path}: shard index does not list exactly its {written} written chunks.")
            if np.any(offsets + sizes > data_bytes):
                raise ValueError(f"{path}: a shard index entry points past the chunk data.")
            if layout.codec == "raw" and not (
                np.array_equal(offsets, np.arange(written) * layout.inner_chunk_bytes)
                and bool(np.all(sizes == layout.inner_chunk_bytes))
                and data_bytes == written * layout.inner_chunk_bytes
            ):
                raise ValueError(f"{path}: raw inner chunks are not whole and in row order.")
            if layout.codec == "zstd" and not (
                int(offsets[0]) == 0
                and np.array_equal(offsets[1:], offsets[:-1] + sizes[:-1])
                and data_bytes == int(offsets[-1] + sizes[-1])
            ):
                raise ValueError(f"{path}: zstd inner chunks are not stored back to back in row order.")
        except BaseException:
            os.close(descriptor)
            raise
        return _OpenShard(descriptor=descriptor, chunk_offsets=offsets, chunk_sizes=sizes, data_bytes=data_bytes)

    def _shard(self, shard_index: int) -> _OpenShard:
        with self._lock:
            shard = self._open_shards.get(shard_index)
            if shard is None:
                shard = self._open_shard(shard_index)
                self._open_shards[shard_index] = shard
            return shard

    def shard_pieces(self, row_start: int, row_stop: int) -> Iterator[tuple[int, int, int]]:
        """Yield (shard_index, local_start, local_stop) covering rows [row_start, row_stop)."""
        if not 0 <= row_start <= row_stop <= self.layout.row_count:
            raise IndexError(f"rows [{row_start}, {row_stop}) fall outside [0, {self.layout.row_count}).")
        shard_rows = self.layout.shard_rows
        cursor = row_start
        while cursor < row_stop:
            shard_index = cursor // shard_rows
            shard_stop = min(row_stop, (shard_index + 1) * shard_rows)
            yield shard_index, cursor - shard_index * shard_rows, shard_stop - shard_index * shard_rows
            cursor = shard_stop

    def viewable(self, row_start: int, row_stop: int) -> bool:
        """Whether rows can be returned as a zero-copy view (raw codec, one shard)."""
        return self.layout.codec == "raw" and len(list(self.shard_pieces(row_start, row_stop))) == 1

    def _mapping(self, shard_index: int) -> mmap.mmap:
        shard = self._shard(shard_index)
        with self._lock:
            if shard.mapping is None:
                shard.mapping = mmap.mmap(shard.descriptor, shard.data_bytes, access=mmap.ACCESS_READ)
            return shard.mapping

    def advise_rows(self, row_start: int, row_stop: int) -> None:
        """Start asynchronous read-ahead of raw rows into the page cache (``POSIX_FADV_WILLNEED``)."""
        sample_count = self.layout.sample_count
        for shard_index, local_start, local_stop in self.shard_pieces(row_start, row_stop):
            byte_start = local_start * sample_count
            os.posix_fadvise(
                self._shard(shard_index).descriptor, byte_start, local_stop * sample_count - byte_start, os.POSIX_FADV_WILLNEED
            )

    def view_rows(self, row_start: int, row_stop: int) -> U8Array:
        """Zero-copy read-only view of raw rows that lie in a single shard."""
        if not self.viewable(row_start, row_stop):
            raise ValueError(f"rows [{row_start}, {row_stop}) of {self.directory} cannot be viewed without a copy.")
        shard_index, local_start, local_stop = next(self.shard_pieces(row_start, row_stop))
        sample_count = self.layout.sample_count
        flat = np.frombuffer(
            self._mapping(shard_index),
            dtype=np.uint8,
            count=(local_stop - local_start) * sample_count,
            offset=local_start * sample_count,
        )
        return flat.reshape(local_stop - local_start, sample_count)

    def _decode_frame(self, frame: memoryview, destination: memoryview, chunk: int) -> None:
        """Decode one inner-chunk frame into ``destination``, which it must fill exactly."""
        reader = _decompressor().stream_reader(frame)
        filled = 0
        while filled < destination.nbytes:
            got = reader.readinto(destination[filled:])
            if got == 0:
                raise ValueError(f"{self.directory}: inner chunk {chunk} decodes to {filled} bytes.")
            filled += got
        if reader.read(1):
            raise ValueError(f"{self.directory}: inner chunk {chunk} decodes to more than {filled} bytes.")

    def _decode_rows_into(self, shard: _OpenShard, local_start: int, local_stop: int, target: U8Array) -> None:
        """Decode zstd rows [local_start, local_stop) of one shard into ``target``.

        The frames covering the rows sit back to back, so one preadv reads them all into this
        thread's buffer; each frame's crc32c is checked, and a frame whose rows are all wanted
        decodes straight into ``target``'s rows instead of through the chunk scratch.
        """
        layout = self.layout
        first_chunk = local_start // layout.inner_rows
        stop_chunk = -(-local_stop // layout.inner_rows)
        begin = int(shard.chunk_offsets[first_chunk])
        end = int(shard.chunk_offsets[stop_chunk - 1] + shard.chunk_sizes[stop_chunk - 1])
        encoded = memoryview(_scratch("encoded_frames", end - begin))
        _pread_exact(shard.descriptor, [encoded], begin)
        for chunk in range(first_chunk, stop_chunk):
            frame_start = int(shard.chunk_offsets[chunk]) - begin
            frame_stop = frame_start + int(shard.chunk_sizes[chunk]) - _CRC32C_BYTES
            frame = encoded[frame_start:frame_stop]
            # google_crc32c takes read-only bytes only, so the compressed frame is copied once.
            if google_crc32c.value(bytes(frame)) != int.from_bytes(encoded[frame_stop : frame_stop + _CRC32C_BYTES], "little"):
                raise ValueError(f"{self.directory}: inner chunk {chunk} fails its crc32c check.")
            chunk_start = chunk * layout.inner_rows
            first, last = max(local_start, chunk_start), min(local_stop, chunk_start + layout.inner_rows)
            rows = target[first - local_start : last - local_start]
            if last - first == layout.inner_rows and rows.flags.c_contiguous:
                self._decode_frame(frame, memoryview(rows.reshape(-1)), chunk)
                continue
            decoded = _scratch("decoded_chunk", layout.inner_chunk_bytes)
            self._decode_frame(frame, memoryview(decoded), chunk)
            rows[...] = decoded.reshape(layout.inner_rows, layout.sample_count)[first - chunk_start : last - chunk_start]

    def read_rows_into(self, row_start: int, row_stop: int, out: U8Array) -> None:
        """Fill ``out`` [rows, samples] (each row contiguous, e.g. a column slice of a wider array)."""
        layout = self.layout
        if out.shape != (row_stop - row_start, layout.sample_count) or out.dtype != np.uint8 or out.strides[1] != 1:
            raise ValueError(f"out must be uint8 [{row_stop - row_start}, {layout.sample_count}] with contiguous rows.")
        out_row = 0
        for shard_index, local_start, local_stop in self.shard_pieces(row_start, row_stop):
            shard = self._shard(shard_index)
            if layout.codec == "raw":
                target = out[out_row : out_row + local_stop - local_start]
                _pread_exact(shard.descriptor, _row_buffers(target), local_start * layout.sample_count)
            else:
                self._decode_rows_into(shard, local_start, local_stop, out[out_row : out_row + local_stop - local_start])
            out_row += local_stop - local_start

    @property
    def shard_count(self) -> int:
        return self.layout.shard_count

    def close(self) -> None:
        """Close the shard descriptors; a mapping lives on until its last view is released."""
        with self._lock:
            for shard in self._open_shards.values():
                os.close(shard.descriptor)
            self._open_shards.clear()


def _column_metadata(dtype: np.dtype[Any], length: int, attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [length],
        "data_type": _ZARR_DATA_TYPES[dtype],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [length]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": False if dtype == np.bool_ else 0,
        "codecs": [_LITTLE_ENDIAN_BYTES_CODEC],
        "attributes": dict(attributes),
    }


def create_column(
    directory: Path,
    dtype: np.dtype[Any] | type,
    length: int,
    attributes: Mapping[str, Any] | None = None,
) -> np.memmap[Any, Any]:
    """Create a 1-D single-chunk Zarr v3 column and return a writable memmap of its values."""
    column_dtype = np.dtype(dtype)
    if column_dtype not in _ZARR_DATA_TYPES or length < 1:
        raise ValueError(f"unsupported column dtype {column_dtype} or length {length}.")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _ZARR_METADATA_FILE).write_text(json.dumps(_column_metadata(column_dtype, length, attributes or {}), indent=1))
    chunk_path = directory / "c" / "0"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(chunk_path, dtype=column_dtype.newbyteorder("<"), mode="w+", shape=(length,))


def write_column(directory: Path, values: NDArray, attributes: Mapping[str, Any] | None = None) -> None:
    column = create_column(directory, values.dtype, values.shape[0], attributes)
    column[:] = values
    column.flush()


def open_column(directory: Path, *, writable: bool = False) -> tuple[np.memmap[Any, Any], dict[str, Any]]:
    """Memory-map a column written by :func:`create_column`; return (values, attributes)."""
    metadata = _read_metadata(directory)
    _require(metadata.get("zarr_format") == 3 and len(metadata["shape"]) == 1, directory, "not a 1-D Zarr v3 column")
    _require(metadata["codecs"] == [_LITTLE_ENDIAN_BYTES_CODEC], directory, "column must be raw little-endian")
    length = int(metadata["shape"][0])
    _require(metadata["chunk_grid"]["configuration"]["chunk_shape"] == [length], directory, "column must be one chunk")
    dtype = _NUMPY_DATA_TYPES[metadata["data_type"]].newbyteorder("<")
    values = np.memmap(directory / "c" / "0", dtype=dtype, mode="r+" if writable else "r", shape=(length,))
    return values, dict(metadata.get("attributes", {}))


# ---------------------------------------------------------------------------
# Manifest, variant table and statistics sidecar
# ---------------------------------------------------------------------------


def dosage_array_directory(root: Path, half_index: int, chromosome: str) -> Path:
    return root / "dosage" / f"half{half_index}" / chromosome


def variant_column_directory(root: Path, chromosome: str, column: str) -> Path:
    return root / "variants" / chromosome / column


def statistic_column_directory(root: Path, half_index: int, chromosome: str, column: str) -> Path:
    return root / "stats" / f"half{half_index}" / chromosome / column


def chromosome_number(chromosome: str) -> int:
    """Autosome number of a 'chrK' store chromosome."""
    if not chromosome.startswith("chr") or not chromosome[3:].isdigit() or not 1 <= int(chromosome[3:]) <= 22:
        raise ValueError(f"store chromosomes are chr1..chr22; got {chromosome!r}.")
    return int(chromosome[3:])


def sites_md5(positions: NDArray, reference_lengths: NDArray, alternate_lengths: NDArray) -> str:
    """md5 of the little-endian int64 (pos, ref_len, alt_len) triples of one chromosome."""
    triples = np.stack([np.asarray(values, dtype="<i8") for values in (positions, reference_lengths, alternate_lengths)], axis=1)
    return hashlib.md5(triples.tobytes()).hexdigest()


def write_manifest(
    root: Path,
    *,
    chromosomes: Sequence[str],
    record_counts: Sequence[int],
    half_sample_counts: Sequence[int],
    chromosome_sites_md5: Sequence[str],
    attributes: Mapping[str, Any] | None = None,
) -> None:
    if not chromosomes or not half_sample_counts or not len(chromosomes) == len(record_counts) == len(chromosome_sites_md5):
        raise ValueError("a store needs matching chromosome, record-count and md5 lists and at least one half.")
    if min(record_counts) < 1 or min(half_sample_counts) < 1:
        raise ValueError("every chromosome needs records and every half needs samples.")
    for chromosome in chromosomes:
        chromosome_number(chromosome)
    manifest = {
        "format": STORE_FORMAT,
        "chromosomes": list(chromosomes),
        "record_counts": [int(count) for count in record_counts],
        "half_sample_counts": [int(count) for count in half_sample_counts],
        "sites_md5": dict(zip(chromosomes, chromosome_sites_md5)),
        "attributes": dict(attributes or {}),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1))


def read_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST_FILE).read_text())
    if manifest.get("format") != STORE_FORMAT:
        raise ValueError(f"{root} is not an {STORE_FORMAT} store (format={manifest.get('format')!r}).")
    return manifest


def write_variant_ids(root: Path, chromosome: str, variant_ids: Sequence[str]) -> None:
    encoded = [variant_id.encode() for variant_id in variant_ids]
    offsets = np.zeros(len(encoded) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum([len(variant_id) for variant_id in encoded], dtype=np.uint64)
    write_column(variant_column_directory(root, chromosome, _ID_BYTES_COLUMN), np.frombuffer(b"".join(encoded), dtype=np.uint8))
    write_column(variant_column_directory(root, chromosome, _ID_OFFSETS_COLUMN), offsets)


@dataclass(frozen=True, slots=True)
class VariantTable:
    """One row per store record, in store order over all chromosomes; no per-variant objects.

    ``variant_class`` indexes ``VARIANT_CLASSES`` (``tuple(VariantClass)``).  ``group_first`` is
    the first row of the record's unbreakable group (its bubble, same-POS set, duplicate group
    or TR locus; its own row if none).  ``sum_code``/``sum_code2`` are the sidecar's all-sample
    code sums over the store's halves, for pre-filtering only.  ``annotations`` holds every
    other sidecar column: categorical and boolean ones as int32 codes whose names are in
    ``annotation_legends``, the rest as float64.
    """

    chromosome: NDArray
    position: I64Array
    genetic_position_cm: F64Array
    ref_length: NDArray
    alt_length: NDArray
    variant_class: U8Array
    group_first: I64Array
    sum_code: NDArray
    sum_code2: NDArray
    annotations: dict[str, NDArray]
    annotation_legends: dict[str, tuple[str, ...]]
    id_bytes: U8Array
    id_offsets: I64Array

    @property
    def variant_count(self) -> int:
        return int(self.position.shape[0])

    def variant_ids(self, rows: NDArray) -> list[str]:
        buffer = self.id_bytes.tobytes()
        return [
            buffer[int(self.id_offsets[row]) : int(self.id_offsets[row + 1])].decode()
            for row in np.asarray(rows, dtype=np.int64)
        ]


def _read_variant_table(root: Path, manifest: Mapping[str, Any], half_indices: Sequence[int]) -> VariantTable:
    chromosomes = manifest["chromosomes"]
    record_counts = manifest["record_counts"]
    reserved = set(REQUIRED_VARIANT_COLUMNS) | {_ID_BYTES_COLUMN, _ID_OFFSETS_COLUMN}
    annotation_names: list[str] | None = None
    parts: dict[str, list[NDArray]] = {}
    legends: dict[str, tuple[str, ...]] = {}
    chromosome_start = 0
    byte_start = 0
    for chromosome, record_count in zip(chromosomes, record_counts):
        present = sorted(path.name for path in (root / "variants" / chromosome).iterdir())
        missing = sorted(reserved - set(present))
        if missing:
            raise ValueError(f"variant table of {chromosome} lacks required columns {missing}.")
        names = [name for name in present if name not in reserved]
        if annotation_names is None:
            annotation_names = names
        elif names != annotation_names:
            raise ValueError(f"annotation columns of {chromosome} differ from those of {chromosomes[0]}.")
        columns: dict[str, NDArray] = {}
        for name in [*REQUIRED_VARIANT_COLUMNS, *names]:
            values, attributes = open_column(variant_column_directory(root, chromosome, name))
            if values.shape[0] != record_count:
                raise ValueError(f"variant column {chromosome}/{name} has {values.shape[0]} rows, not {record_count}.")
            columns[name] = values
            if name in names and (values.dtype == np.bool_ or "legend" in attributes):
                legend = tuple(attributes["legend"]) if "legend" in attributes else ("false", "true")
                if legends.setdefault(name, legend) != legend:
                    raise ValueError(f"annotation {name} changes its legend at {chromosome}.")
        if sites_md5(columns["pos"], columns["ref_len"], columns["alt_len"]) != manifest["sites_md5"][chromosome]:
            raise ValueError(f"{chromosome} sites (pos, ref_len, alt_len) do not match the manifest md5.")
        if int(columns["variant_class"].max()) >= len(VARIANT_CLASSES):
            raise ValueError(f"{chromosome} has variant_class codes outside tuple(VariantClass).")
        group_first = columns["group_first"].astype(np.int64)
        if np.any(group_first > np.arange(record_count)) or np.any(group_first < 0):
            raise ValueError(f"{chromosome} group_first must point at or before each row.")
        ids, _ = open_column(variant_column_directory(root, chromosome, _ID_BYTES_COLUMN))
        offsets, _ = open_column(variant_column_directory(root, chromosome, _ID_OFFSETS_COLUMN))
        if offsets.shape[0] != record_count + 1 or int(offsets[-1]) != ids.shape[0]:
            raise ValueError(f"variant ids of {chromosome} do not match its {record_count} records.")
        half_totals = []
        for statistic in ("sum_code", "sum_code2"):
            total = np.zeros(record_count, dtype=np.uint64)
            for half in half_indices:
                values, _ = open_column(statistic_column_directory(root, half, chromosome, statistic))
                if values.shape[0] != record_count:
                    raise ValueError(f"statistic {statistic} of half{half}/{chromosome} has {values.shape[0]} rows.")
                total += values.astype(np.uint64)
            half_totals.append(total)
        chromosome_parts = {
            "chromosome": np.full(record_count, chromosome_number(chromosome), dtype=np.int8),
            "position": columns["pos"].astype(np.int64),
            "genetic_position_cm": columns["cm"].astype(np.float64),
            "ref_length": columns["ref_len"].astype(np.int32),
            "alt_length": columns["alt_len"].astype(np.int32),
            "variant_class": columns["variant_class"].astype(np.uint8),
            "group_first": group_first + chromosome_start,
            "sum_code": half_totals[0],
            "sum_code2": half_totals[1],
            "id_bytes": np.asarray(ids),
            "id_offsets": offsets[:-1].astype(np.int64) + byte_start,
        }
        for name in names:
            chromosome_parts["annotation:" + name] = (
                columns[name].astype(np.int32) if name in legends else columns[name].astype(np.float64)
            )
        for name, values in chromosome_parts.items():
            parts.setdefault(name, []).append(values)
        chromosome_start += record_count
        byte_start += int(ids.shape[0])
    parts["id_offsets"].append(np.array([byte_start], dtype=np.int64))
    merged = {name: np.concatenate(values) for name, values in parts.items()}
    return VariantTable(
        chromosome=merged["chromosome"],
        position=merged["position"],
        genetic_position_cm=merged["genetic_position_cm"],
        ref_length=merged["ref_length"],
        alt_length=merged["alt_length"],
        variant_class=merged["variant_class"],
        group_first=merged["group_first"],
        sum_code=merged["sum_code"],
        sum_code2=merged["sum_code2"],
        annotations={name.split(":", 1)[1]: values for name, values in merged.items() if name.startswith("annotation:")},
        annotation_legends=legends,
        id_bytes=merged["id_bytes"],
        id_offsets=merged["id_offsets"],
    )


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def _ensure_open_file_capacity(required_descriptors: int) -> None:
    """Raise the soft open-file limit to cover one descriptor and one mapping per shard."""
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit < required_descriptors:
        if hard_limit != resource.RLIM_INFINITY and hard_limit < required_descriptors:
            raise RuntimeError(f"the store needs {required_descriptors} open files but the hard limit is {hard_limit}.")
        resource.setrlimit(resource.RLIMIT_NOFILE, (required_descriptors, hard_limit))


@dataclass(frozen=True, slots=True)
class _SampleSelection:
    """Store columns a read returns: all of them, one contiguous range, or a sorted gather."""

    indices: I64Array
    contiguous: bool
    complete: bool

    @classmethod
    def build(cls, sample_indices: NDArray | None, sample_count: int) -> _SampleSelection:
        if sample_indices is None:
            return cls(indices=np.arange(sample_count, dtype=np.int64), contiguous=True, complete=True)
        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= sample_count:
            raise ValueError("sample_indices must be sorted, distinct store columns.")
        contiguous = bool(indices[-1] - indices[0] + 1 == indices.size)
        return cls(indices=indices, contiguous=contiguous, complete=contiguous and indices.size == sample_count)


class DosageStore:
    """An svpgs-store v1 opened for reading.

    The variant axis concatenates the chromosomes in manifest order and the sample axis the
    selected halves (all halves unless ``half_indices`` names a subset).
    """

    def __init__(self, root: Path, half_indices: Sequence[int] | None = None) -> None:
        self.root = Path(root)
        manifest = read_manifest(self.root)
        self.chromosomes: tuple[str, ...] = tuple(manifest["chromosomes"])
        self.record_counts: tuple[int, ...] = tuple(manifest["record_counts"])
        all_half_counts = tuple(manifest["half_sample_counts"])
        selected = tuple(range(len(all_half_counts))) if half_indices is None else tuple(half_indices)
        if not selected or len(set(selected)) != len(selected) or not all(0 <= half < len(all_half_counts) for half in selected):
            raise ValueError(f"half_indices {half_indices} must be distinct indices below {len(all_half_counts)}.")
        self.half_indices = selected
        self.manifest_attributes: dict[str, Any] = manifest["attributes"]
        self.chromosome_starts = np.concatenate([[0], np.cumsum(self.record_counts)]).astype(np.int64)
        half_counts = [all_half_counts[half] for half in selected]
        self.half_sample_starts = np.concatenate([[0], np.cumsum(half_counts)]).astype(np.int64)
        self._arrays: list[list[CodeArray]] = []
        for half, sample_count in zip(selected, half_counts):
            arrays = [CodeArray(dosage_array_directory(self.root, half, chromosome)) for chromosome in self.chromosomes]
            for array, record_count in zip(arrays, self.record_counts):
                if (array.layout.row_count, array.layout.sample_count) != (record_count, sample_count):
                    raise ValueError(f"{array.directory} does not match the manifest's {record_count} x {sample_count}.")
            self._arrays.append(arrays)
        _ensure_open_file_capacity(2 * sum(array.shard_count for arrays in self._arrays for array in arrays) + 256)
        self.variant_table = _read_variant_table(self.root, manifest, selected)
        self.reader_threads = len(os.sched_getaffinity(0))
        self._pool = ThreadPoolExecutor(max_workers=self.reader_threads, thread_name_prefix="dosage-store")

    @classmethod
    def open(cls, path: str | Path, half_indices: Sequence[int] | None = None) -> DosageStore:
        return cls(Path(path), half_indices)

    @property
    def n_variants(self) -> int:
        return int(self.chromosome_starts[-1])

    @property
    def n_samples(self) -> int:
        return int(self.half_sample_starts[-1])

    def statistic(self, name: str) -> I64Array:
        """A per-record integer sidecar statistic summed over the store's halves, in store order."""
        per_chromosome = []
        for chromosome, record_count in zip(self.chromosomes, self.record_counts):
            total = np.zeros(record_count, dtype=np.int64)
            for half in self.half_indices:
                values, _ = open_column(statistic_column_directory(self.root, half, chromosome, name))
                total += values.astype(np.int64)
            per_chromosome.append(total)
        return np.concatenate(per_chromosome)

    def _chromosome_pieces(self, start: int, stop: int) -> Iterator[tuple[int, int, int, int]]:
        """Yield (chromosome_index, local_start, local_stop, out_row) covering a global row range."""
        if not 0 <= start <= stop <= self.n_variants:
            raise IndexError(f"rows [{start}, {stop}) fall outside [0, {self.n_variants}).")
        cursor = start
        while cursor < stop:
            chromosome = int(np.searchsorted(self.chromosome_starts, cursor, side="right")) - 1
            chromosome_start = int(self.chromosome_starts[chromosome])
            piece_stop = min(stop, int(self.chromosome_starts[chromosome + 1]))
            yield chromosome, cursor - chromosome_start, piece_stop - chromosome_start, cursor - start
            cursor = piece_stop

    def _viewable(self, start: int, stop: int, selection: _SampleSelection) -> bool:
        if not selection.complete or len(self._arrays) != 1:
            return False
        pieces = list(self._chromosome_pieces(start, stop))
        return len(pieces) == 1 and self._arrays[0][pieces[0][0]].viewable(pieces[0][1], pieces[0][2])

    def _fill_piece(
        self,
        array: CodeArray,
        local_start: int,
        local_stop: int,
        half_columns: I64Array,
        destination: U8Array,
    ) -> None:
        """Read rows of one half into ``destination``, taking ``half_columns`` of that half."""
        if half_columns.size == array.layout.sample_count:
            array.read_rows_into(local_start, local_stop, destination)
            return
        if array.viewable(local_start, local_stop):
            full = array.view_rows(local_start, local_stop)
        else:
            full = _scratch("gather_rows", (local_stop - local_start) * array.layout.sample_count).reshape(
                local_stop - local_start, array.layout.sample_count
            )
            array.read_rows_into(local_start, local_stop, full)
        if half_columns[-1] - half_columns[0] + 1 == half_columns.size:
            destination[...] = full[:, half_columns[0] : half_columns[-1] + 1]
        else:
            # The columns were validated by _SampleSelection, so "clip" never clips; it only
            # lets take write straight into the strided destination without buffering.
            np.take(full, half_columns, axis=1, out=destination, mode="clip")

    def _read_into(self, start: int, stop: int, selection: _SampleSelection, out: U8Array) -> None:
        tasks = []
        output_column = 0
        for half_position, arrays in enumerate(self._arrays):
            half_start = int(self.half_sample_starts[half_position])
            half_stop = int(self.half_sample_starts[half_position + 1])
            lower, upper = np.searchsorted(selection.indices, [half_start, half_stop])
            if lower == upper:
                continue
            half_columns = selection.indices[lower:upper] - half_start
            output_columns = slice(output_column, output_column + int(upper - lower))
            output_column += int(upper - lower)
            for chromosome, local_start, local_stop, out_row in self._chromosome_pieces(start, stop):
                array = arrays[chromosome]
                inner_rows = array.layout.inner_rows
                step = -(-max(inner_rows, -(-(local_stop - local_start) // self.reader_threads)) // inner_rows) * inner_rows
                cuts = [local_start, *range((local_start // step + 1) * step, local_stop, step), local_stop]
                for piece_start, piece_stop in zip(cuts[:-1], cuts[1:]):
                    rows = slice(out_row + piece_start - local_start, out_row + piece_stop - local_start)
                    tasks.append(
                        self._pool.submit(self._fill_piece, array, piece_start, piece_stop, half_columns, out[rows, output_columns])
                    )
        for task in tasks:
            task.result()

    def read_codes(
        self,
        start: int,
        stop: int,
        sample_indices: NDArray | None = None,
        out: U8Array | None = None,
    ) -> U8Array:
        """Codes [stop - start, selected samples], C-contiguous.

        With every sample selected, no ``out``, one half and a raw range inside one shard, the
        result is a zero-copy view of the page cache.  Otherwise the codes are read (in parallel)
        into ``out``, or into a new array.
        """
        selection = _SampleSelection.build(sample_indices, self.n_samples)
        rows = stop - start
        if out is None:
            if self._viewable(start, stop, selection):
                pieces = next(self._chromosome_pieces(start, stop))
                return self._arrays[0][pieces[0]].view_rows(pieces[1], pieces[2])
            out = np.empty((rows, selection.indices.size), dtype=np.uint8)
        if out.dtype != np.uint8 or out.shape != (rows, selection.indices.size) or not out.flags.c_contiguous:
            raise ValueError(f"out must be C-contiguous uint8 [{rows}, {selection.indices.size}].")
        self._read_into(start, stop, selection, out)
        return out

    def advise(self, start: int, stop: int) -> None:
        """Start asynchronous page-cache read-ahead of a raw row range in every selected half."""
        for chromosome, local_start, local_stop, _ in self._chromosome_pieces(start, stop):
            for arrays in self._arrays:
                if arrays[chromosome].layout.codec == "raw":
                    arrays[chromosome].advise_rows(local_start, local_stop)

    def iter_codes(
        self,
        row_ranges: Sequence[tuple[int, int]],
        sample_indices: NDArray | None,
        budget: ComputeBudget,
    ) -> Iterator[tuple[int, int, U8Array]]:
        """Yield (start, stop, codes) for each range in order, reading ahead in the background.

        On a CPU budget with every sample of a one-half raw store selected, this is the large-RAM
        path: each range is a zero-copy page-cache view where it sits inside one shard (else it
        is read into one spare buffer), and the next range is advised for read-ahead.  Every other
        case streams through a ring of buffers filled by a background reader.  The ring is
        pinned host memory on a CUDA budget, and 2 to 3 buffers deep, as ``budget.host_bytes``
        allows.  A yielded block stays valid until the caller asks for the next one.
        """
        selection = _SampleSelection.build(sample_indices, self.n_samples)
        ranges = [(int(start), int(stop)) for start, stop in row_ranges]
        if not ranges:
            return
        raw_arrays = all(array.layout.codec == "raw" for arrays in self._arrays for array in arrays)
        if budget.device_kind == "cpu" and selection.complete and len(self._arrays) == 1 and raw_arrays:
            spare = np.empty(0, dtype=np.uint8)
            for position, (start, stop) in enumerate(ranges):
                if position + 1 < len(ranges):
                    self.advise(*ranges[position + 1])
                if self._viewable(start, stop, selection):
                    yield start, stop, self.read_codes(start, stop)
                    continue
                if spare.size < (stop - start) * self.n_samples:
                    spare = np.empty((stop - start) * self.n_samples, dtype=np.uint8)
                yield start, stop, self.read_codes(start, stop, None, spare[: (stop - start) * self.n_samples].reshape(stop - start, -1))
            return
        widest = max(stop - start for start, stop in ranges)
        buffer_bytes = widest * selection.indices.size
        depth = min(3, budget.host_bytes // max(buffer_bytes, 1))
        if depth < 2:
            raise MemoryError(f"double-buffering {widest}-row blocks needs {2 * buffer_bytes} host bytes.")
        ring, pinned = self._ring(depth, buffer_bytes, budget)
        # Every ring slot but the consumer's is read concurrently: one block has only as many
        # independent pieces as it has inner chunks, so a lone reader leaves most of the pool idle
        # while it decodes (zstd) or waits (network storage).
        prefetch = ThreadPoolExecutor(max_workers=depth - 1, thread_name_prefix="dosage-store-prefetch")
        in_flight: deque[Future[None]] = deque()

        def submit(position: int) -> None:
            start, stop = ranges[position]
            target = ring[position % depth][: (stop - start) * selection.indices.size].reshape(stop - start, -1)
            in_flight.append(prefetch.submit(self._read_into, start, stop, selection, target))

        try:
            for position in range(min(depth - 1, len(ranges))):
                submit(position)
            for position, (start, stop) in enumerate(ranges):
                in_flight.popleft().result()
                if position + depth - 1 < len(ranges):
                    submit(position + depth - 1)
                yield start, stop, ring[position % depth][: (stop - start) * selection.indices.size].reshape(stop - start, -1)
        finally:
            for pending in in_flight:
                pending.cancel()
            prefetch.shutdown(wait=True)
            for owner in pinned:
                _release_pinned(owner)

    def _ring(self, depth: int, buffer_bytes: int, budget: ComputeBudget) -> tuple[list[U8Array], list[Any]]:
        if budget.device_kind == "cpu":
            return [np.empty(buffer_bytes, dtype=np.uint8) for _ in range(depth)], []
        cupy = _try_import_cupy()
        if cupy is None:
            raise RuntimeError("a CUDA budget needs CuPy to pin the dosage staging buffers.")
        owners, buffers = [], []
        for _ in range(depth):
            owner, view = _allocate_pinned(cupy, buffer_bytes)
            owners.append(owner)
            buffers.append(view[:buffer_bytes])
        return buffers, owners

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for arrays in self._arrays:
            for array in arrays:
                array.close()

    def __enter__(self) -> DosageStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def write_dosage_store(
    path: str | Path,
    n_samples: int,
    variant_table: VariantTable,
    code_blocks: Iterable[U8Array],
    *,
    codec: Codec,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    inner_rows: int = DEFAULT_INNER_CHUNK_ROWS,
) -> None:
    """Write a one-half store from code blocks [rows, n_samples] that arrive in store order.

    The sidecar code sums are computed from the blocks and must equal the table's
    ``sum_code``/``sum_code2``.
    """
    root = Path(path)
    table = variant_table
    boundaries = np.flatnonzero(np.diff(table.chromosome.astype(np.int64))) + 1
    starts = np.concatenate([[0], boundaries]).astype(np.int64)
    stops = np.concatenate([boundaries, [table.variant_count]]).astype(np.int64)
    if np.any(np.diff(table.chromosome[starts].astype(np.int64)) <= 0):
        raise ValueError("variant_table rows must be grouped by ascending chromosome.")
    chromosomes = [f"chr{int(table.chromosome[start])}" for start in starts]
    written_sums = np.zeros(table.variant_count, dtype=np.uint64)
    written_squares = np.zeros(table.variant_count, dtype=np.uint64)
    blocks = iter(code_blocks)
    pending = np.empty((0, n_samples), dtype=np.uint8)
    for chromosome, chromosome_start, chromosome_stop in zip(chromosomes, starts.tolist(), stops.tolist()):
        record_count = chromosome_stop - chromosome_start
        rows = slice(chromosome_start, chromosome_stop)
        for name, values, attributes in (
            ("pos", table.position[rows], {}),
            ("ref_len", table.ref_length[rows].astype(np.int32), {}),
            ("alt_len", table.alt_length[rows].astype(np.int32), {}),
            ("cm", table.genetic_position_cm[rows], {}),
            ("variant_class", table.variant_class[rows], {"legend": [variant_class.value for variant_class in VARIANT_CLASSES]}),
            ("group_first", table.group_first[rows] - chromosome_start, {}),
        ):
            write_column(variant_column_directory(root, chromosome, name), values, attributes)
        for name, values in table.annotations.items():
            legend = table.annotation_legends.get(name)
            write_column(
                variant_column_directory(root, chromosome, name),
                values[rows],
                {} if legend is None else {"legend": list(legend)},
            )
        write_variant_ids(root, chromosome, table.variant_ids(np.arange(chromosome_start, chromosome_stop)))
        directory = dosage_array_directory(root, 0, chromosome)
        layout = create_code_array(directory, record_count, n_samples, codec=codec, shard_rows=shard_rows, inner_rows=inner_rows)
        written = 0
        for shard_index in range(layout.shard_count):
            with CodeShardWriter(directory, layout, shard_index) as writer:
                remaining = layout.shard_row_count(shard_index)
                while remaining:
                    if pending.shape[0] == 0:
                        block = next(blocks, None)
                        if block is None:
                            raise ValueError("code_blocks hold fewer rows than the variant table.")
                        pending = block
                    take = min(remaining, pending.shape[0])
                    writer.write_rows(pending[:take])
                    block_rows = slice(chromosome_start + written, chromosome_start + written + take)
                    written_sums[block_rows], written_squares[block_rows] = code_sums(pending[:take])
                    pending = pending[take:]
                    written += take
                    remaining -= take
        write_column(statistic_column_directory(root, 0, chromosome, "sum_code"), written_sums[rows])
        write_column(statistic_column_directory(root, 0, chromosome, "sum_code2"), written_squares[rows])
    if pending.shape[0] or next(blocks, None) is not None:
        raise ValueError("code_blocks hold more rows than the variant table.")
    if not (np.array_equal(written_sums, table.sum_code) and np.array_equal(written_squares, table.sum_code2)):
        raise ValueError("the variant table's sum_code/sum_code2 disagree with the code blocks.")
    write_manifest(
        root,
        chromosomes=chromosomes,
        record_counts=(stops - starts).tolist(),
        half_sample_counts=[n_samples],
        chromosome_sites_md5=[
            sites_md5(table.position[start:stop], table.ref_length[start:stop], table.alt_length[start:stop])
            for start, stop in zip(starts.tolist(), stops.tolist())
        ],
    )


def transcode_store(store: DosageStore, destination: str | Path, *, codec: Codec, budget: ComputeBudget) -> None:
    """Rewrite a store's selected halves as one half with ``codec``.

    This builds the local cache: bucket store (zstd, several halves) to one raw half on
    NVMe or in RAM, whose ranges are zero-copy views.
    """
    ranges = _block_ranges(0, store.n_variants, store.n_samples, budget.host_bytes // 8)
    blocks = (block for _, _, block in store.iter_codes(ranges, None, budget))
    write_dosage_store(destination, store.n_samples, store.variant_table, blocks, codec=codec)


def code_sums(codes: U8Array) -> tuple[I64Array, I64Array]:
    """Exact per-row sums of code and code**2 (code**2 <= 64516 fits uint16)."""
    squares = codes.astype(np.uint16)
    np.multiply(squares, squares, out=squares)
    return np.add.reduce(codes, axis=1, dtype=np.int64), np.add.reduce(squares, axis=1, dtype=np.int64)


def _block_ranges(start: int, stop: int, bytes_per_row: int, budget_bytes: int) -> list[tuple[int, int]]:
    """Contiguous row ranges whose working set, at ``bytes_per_row``, fits ``budget_bytes``."""
    rows = max(1, budget_bytes // max(bytes_per_row, 1))
    return [(block_start, min(stop, block_start + rows)) for block_start in range(start, stop, rows)]


class StoreTileSource:
    """A store seen one chromosome at a time: ``genotype_statistics.GenotypeTileSource``.

    Every record of every chromosome streams, in store order; ``unsplittable_groups`` is the
    variant table's ``group_first``, so a block never splits a bubble, same-POS set, duplicate
    group or TR locus.
    """

    def __init__(self, store: DosageStore) -> None:
        self._store = store

    @property
    def sample_count(self) -> int:
        return self._store.n_samples

    def chromosomes(self) -> list[str]:
        return list(self._store.chromosomes)

    def _rows(self, chromosome: str) -> slice:
        index = self._store.chromosomes.index(chromosome)
        return slice(int(self._store.chromosome_starts[index]), int(self._store.chromosome_starts[index + 1]))

    def variant_count(self, chromosome: str) -> int:
        rows = self._rows(chromosome)
        return rows.stop - rows.start

    def unsplittable_groups(self, chromosome: str) -> I64Array:
        return self._store.variant_table.group_first[self._rows(chromosome)]

    def read_rows(self, chromosome: str, start: int, stop: int, out: U8Array) -> None:
        offset = self._rows(chromosome).start
        self._store.read_codes(offset + start, offset + stop, None, out)

    def store_rows(self, chromosome: str) -> I64Array:
        rows = self._rows(chromosome)
        return np.arange(rows.start, rows.stop, dtype=np.int64)

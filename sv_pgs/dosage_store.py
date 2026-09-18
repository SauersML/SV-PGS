"""Quantized dosage store: uint8 ALT-dosage codes in variant-major Zarr v3 shards.

Layout ("svpgs-store v1", target-format REPORT §2 with addenda A1/A2)::

    MANIFEST.json                        format, chromosomes, records per chromosome, samples per half
    dosage/half{h}/{chrom}/              Zarr v3 uint8 [records, samples_h], sharded by record rows
    variants/{chrom}/{column}/           Zarr v3 1-D columns, one row per record
    stats/half{h}/{chrom}/{column}/      exact integer per-record sums over the half's samples

Encoding: ``code = (DS_milli * 127 + 500) // 1000`` in [0, 254] and DS = code / 127, so 0, 1
and 2 are exact and the error is at most 1/254.  Code 255 is the Zarr fill value and is never
written.  Dosages count the ALT allele.

Kernels work on the signed code ``s = code - 127`` in [-127, 127].  Every |s_i s_j| <= 127**2,
so a sum of cross-products over ``INT32_EXACT_ROWS`` rows fits an int32 accumulator and one over
``FLOAT32_EXACT_ROWS`` rows stays an exactly representable fp32 integer.  Standardization is
affine in s, x = (DS - mean_DS) / sd_DS = (s - mu_s) / sigma_s, with mu_s and sigma_s taken from
exact integer sums over the training rows only.

Shards hold their inner chunks raw and in row order, so the rows of one shard are one contiguous
byte range: a range inside a shard is a zero-copy view of the page cache, and any range can be
read with ``preadv`` straight into a caller buffer (pinned host memory for GPU staging).  Variant
metadata is columnar; no Python object is built per variant.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import mmap
import os
from pathlib import Path
import resource
import threading
from typing import Any, Iterator, Mapping, Sequence

import google_crc32c
import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray, U8Array

STORE_FORMAT = "svpgs-store-v1"
MANIFEST_FILE = "MANIFEST.json"
CODES_PER_DOSAGE = 127
MAXIMUM_CODE = 254
MISSING_CODE = 255
SIGNED_CODE_OFFSET = 127
MAXIMUM_DOSAGE_MILLI = 2000
_MAXIMUM_SIGNED_PRODUCT = SIGNED_CODE_OFFSET * SIGNED_CODE_OFFSET
INT32_EXACT_ROWS = (2**31 - 1) // _MAXIMUM_SIGNED_PRODUCT
FLOAT32_EXACT_ROWS = 2**24 // _MAXIMUM_SIGNED_PRODUCT
DEFAULT_SHARD_ROWS = 65536
DEFAULT_INNER_CHUNK_ROWS = 64
REQUIRED_VARIANT_COLUMNS = ("pos", "ref_len", "alt_len")

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


def encode_dosage_milli(dosage_milli: NDArray) -> U8Array:
    """Map integer thousandths of ALT dosage (0..2000) to stored codes (0..254), rounding half up."""
    milli = np.asarray(dosage_milli)
    if milli.dtype.kind not in "iu":
        raise TypeError(f"dosage_milli must be an integer array; got dtype {milli.dtype}.")
    if milli.size and (int(milli.min()) < 0 or int(milli.max()) > MAXIMUM_DOSAGE_MILLI):
        raise ValueError(f"dosage_milli must lie in [0, {MAXIMUM_DOSAGE_MILLI}].")
    return ((milli.astype(np.uint32) * CODES_PER_DOSAGE + 500) // 1000).astype(np.uint8)


def decode_codes(codes: U8Array) -> F64Array:
    """Return the ALT dosage DS = code / 127 that each stored code represents."""
    values = np.asarray(codes)
    if values.dtype != np.uint8:
        raise TypeError(f"codes must be uint8; got dtype {values.dtype}.")
    if values.size and int(values.max()) > MAXIMUM_CODE:
        raise ValueError(f"code {MISSING_CODE} marks a missing dosage and is never stored.")
    return values.astype(np.float64) / CODES_PER_DOSAGE


def signed_codes(codes: U8Array, out: NDArray) -> NDArray:
    """Write s = code - 127 into ``out`` (int8, int16, int32, float32 or float64) and return it.

    For int8 the shift is one wrapping uint8 add: (code + 129) mod 256 read as int8 is code - 127,
    and code 255 never occurs, so s never reaches -128.
    """
    if out.dtype == np.int8:
        np.add(codes, np.uint8(256 - SIGNED_CODE_OFFSET), out=out.view(np.uint8))
    else:
        np.subtract(codes, SIGNED_CODE_OFFSET, out=out, dtype=out.dtype, casting="unsafe")
    return out


# ---------------------------------------------------------------------------
# Zarr v3 containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CodeArrayLayout:
    """Geometry of one sharded uint8 code array [row_count, sample_count]."""

    row_count: int
    sample_count: int
    shard_rows: int
    inner_rows: int

    def __post_init__(self) -> None:
        if self.row_count < 1 or self.sample_count < 1:
            raise ValueError("a code array needs at least one row and one sample.")
        if self.inner_rows < 1 or self.shard_rows % self.inner_rows != 0:
            raise ValueError("shard_rows must be a positive multiple of inner_rows.")

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

    def shard_data_bytes(self, shard_index: int) -> int:
        return self.shard_written_chunks(shard_index) * self.inner_chunk_bytes


def _shard_index_codecs() -> list[dict[str, Any]]:
    return [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]


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
                    "codecs": [{"name": "bytes"}],
                    "index_codecs": _shard_index_codecs(),
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
    inner_codecs = sharding["codecs"]
    _require(len(inner_codecs) == 1 and inner_codecs[0]["name"] == "bytes", directory, "inner chunks must be raw bytes")
    _require(sharding["index_codecs"] == _shard_index_codecs(), directory, "shard index must be little-endian bytes + crc32c")
    _require(sharding["index_location"] == "end", directory, "shard index must sit at the end of the shard")
    return CodeArrayLayout(
        row_count=int(shape[0]),
        sample_count=int(shape[1]),
        shard_rows=int(shard_shape[0]),
        inner_rows=int(inner_shape[0]),
    )


def create_code_array(
    directory: Path,
    row_count: int,
    sample_count: int,
    *,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    inner_rows: int = DEFAULT_INNER_CHUNK_ROWS,
) -> CodeArrayLayout:
    """Write the Zarr v3 metadata of an empty code array and return its layout."""
    layout = CodeArrayLayout(row_count=row_count, sample_count=sample_count, shard_rows=shard_rows, inner_rows=inner_rows)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / _ZARR_METADATA_FILE).write_text(json.dumps(_code_array_metadata(layout), indent=1))
    return layout


def _shard_path(directory: Path, shard_index: int) -> Path:
    return directory / "c" / str(shard_index) / "0"


class CodeShardWriter:
    """Stream the rows of one shard of a code array into its shard file.

    Rows arrive in order; the file is written under a temporary name and renamed into place by
    ``close`` only after every row, the fill padding of the last inner chunk and the checksummed
    index are on disk.  Shards are independent files, so separate processes may write separate
    shards of one array concurrently.
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

    def write_rows(self, codes: U8Array) -> None:
        if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != self._layout.sample_count:
            raise ValueError(f"rows must be uint8 [rows, {self._layout.sample_count}]; got {codes.dtype} {codes.shape}.")
        if self._rows_written + codes.shape[0] > self._rows_expected:
            raise ValueError(f"shard {self._shard_index} holds {self._rows_expected} rows; too many written.")
        if codes.size and int(codes.max()) > MAXIMUM_CODE:
            raise ValueError(f"code {MISSING_CODE} is the missing-dosage fill value and is never stored.")
        self._handle.write(np.ascontiguousarray(codes).reshape(-1).data)
        self._rows_written += codes.shape[0]

    def close(self) -> None:
        if self._rows_written != self._rows_expected:
            raise ValueError(
                f"shard {self._shard_index} received {self._rows_written} of its {self._rows_expected} rows."
            )
        layout = self._layout
        padding_rows = -self._rows_written % layout.inner_rows
        self._handle.write(bytes([MISSING_CODE]) * (padding_rows * layout.sample_count))
        written_chunks = layout.shard_written_chunks(self._shard_index)
        index = np.full((layout.inner_chunks_per_shard, 2), _UNWRITTEN_CHUNK, dtype="<u8")
        index[:written_chunks, 0] = np.arange(written_chunks, dtype=np.uint64) * np.uint64(layout.inner_chunk_bytes)
        index[:written_chunks, 1] = layout.inner_chunk_bytes
        index_bytes = index.tobytes()
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
    data_bytes: int
    mapping: mmap.mmap | None = None


def _io_vector_limit() -> int:
    return int(os.sysconf("SC_IOV_MAX"))


def _pread_exact(descriptor: int, buffers: list[memoryview], offset: int) -> None:
    """Fill every buffer from ``descriptor`` starting at ``offset``; short reads are resumed."""
    vector_limit = _io_vector_limit()
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
            data_bytes = layout.shard_data_bytes(shard_index)
            expected_bytes = data_bytes + layout.shard_index_bytes
            actual_bytes = os.fstat(descriptor).st_size
            if actual_bytes != expected_bytes:
                raise ValueError(f"{path} holds {actual_bytes} bytes; its layout needs {expected_bytes}.")
            index_bytes = os.pread(descriptor, layout.shard_index_bytes, data_bytes)
            index_body = index_bytes[:-_CRC32C_BYTES]
            if google_crc32c.value(index_body) != int.from_bytes(index_bytes[-_CRC32C_BYTES:], "little"):
                raise ValueError(f"{path}: shard index fails its crc32c check.")
            index = np.frombuffer(index_body, dtype="<u8").reshape(layout.inner_chunks_per_shard, 2)
            written_chunks = layout.shard_written_chunks(shard_index)
            expected_offsets = np.arange(written_chunks, dtype=np.uint64) * np.uint64(layout.inner_chunk_bytes)
            contiguous = (
                np.array_equal(index[:written_chunks, 0], expected_offsets)
                and bool(np.all(index[:written_chunks, 1] == layout.inner_chunk_bytes))
                and bool(np.all(index[written_chunks:] == _UNWRITTEN_CHUNK))
            )
            if not contiguous:
                raise ValueError(f"{path}: inner chunks are not stored raw, whole and in row order.")
        except BaseException:
            os.close(descriptor)
            raise
        return _OpenShard(descriptor=descriptor, data_bytes=data_bytes)

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

    def _mapping(self, shard_index: int) -> mmap.mmap:
        shard = self._shard(shard_index)
        with self._lock:
            if shard.mapping is None:
                shard.mapping = mmap.mmap(shard.descriptor, shard.data_bytes, access=mmap.ACCESS_READ)
            return shard.mapping

    def advise_rows(self, row_start: int, row_stop: int) -> None:
        """Ask the kernel to start reading rows into the page cache (``MADV_WILLNEED``)."""
        sample_count = self.layout.sample_count
        for shard_index, local_start, local_stop in self.shard_pieces(row_start, row_stop):
            byte_start = local_start * sample_count // mmap.PAGESIZE * mmap.PAGESIZE
            self._mapping(shard_index).madvise(mmap.MADV_WILLNEED, byte_start, local_stop * sample_count - byte_start)

    def view_rows(self, row_start: int, row_stop: int) -> U8Array:
        """Zero-copy read-only view of rows that lie in a single shard."""
        pieces = list(self.shard_pieces(row_start, row_stop))
        if len(pieces) != 1:
            raise ValueError(f"rows [{row_start}, {row_stop}) span {len(pieces)} shards; a view needs one.")
        shard_index, local_start, local_stop = pieces[0]
        mapping = self._mapping(shard_index)
        sample_count = self.layout.sample_count
        flat = np.frombuffer(
            mapping,
            dtype=np.uint8,
            count=(local_stop - local_start) * sample_count,
            offset=local_start * sample_count,
        )
        return flat.reshape(local_stop - local_start, sample_count)

    def read_rows_into(self, row_start: int, row_stop: int, out: U8Array) -> None:
        """Fill ``out`` [rows, samples] (each row contiguous, e.g. a column slice) with ``preadv``."""
        sample_count = self.layout.sample_count
        if out.shape != (row_stop - row_start, sample_count) or out.dtype != np.uint8 or out.strides[1] != 1:
            raise ValueError(f"out must be uint8 [{row_stop - row_start}, {sample_count}] with contiguous rows.")
        out_row = 0
        for shard_index, local_start, local_stop in self.shard_pieces(row_start, row_stop):
            target = out[out_row : out_row + local_stop - local_start]
            if target.flags.c_contiguous:
                buffers = [memoryview(target.reshape(-1))]
            else:
                buffers = [memoryview(row) for row in target]
            _pread_exact(self._shard(shard_index).descriptor, buffers, local_start * sample_count)
            out_row += local_stop - local_start

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
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": dict(attributes),
    }


def create_column(
    directory: Path,
    dtype: np.dtype[Any] | type,
    length: int,
    attributes: Mapping[str, Any] | None = None,
) -> np.memmap[Any, Any]:
    """Create a 1-D single-chunk Zarr v3 column and return a writable memmap of its values."""
    column_dtype = np.dtype(dtype).newbyteorder("<")
    if column_dtype.newbyteorder("=") not in _ZARR_DATA_TYPES or length < 1:
        raise ValueError(f"unsupported column dtype {column_dtype} or length {length}.")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _column_metadata(column_dtype.newbyteorder("="), length, attributes or {})
    (directory / _ZARR_METADATA_FILE).write_text(json.dumps(metadata, indent=1))
    chunk_path = directory / "c" / "0"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(chunk_path, dtype=column_dtype, mode="w+", shape=(length,))


def write_column(directory: Path, values: NDArray, attributes: Mapping[str, Any] | None = None) -> None:
    column = create_column(directory, values.dtype, values.shape[0], attributes)
    column[:] = values
    column.flush()


def open_column(directory: Path, *, writable: bool = False) -> tuple[np.memmap[Any, Any], dict[str, Any]]:
    """Memory-map a column written by :func:`create_column`; return (values, attributes)."""
    metadata = _read_metadata(directory)
    _require(metadata.get("zarr_format") == 3 and len(metadata["shape"]) == 1, directory, "not a 1-D Zarr v3 column")
    _require(metadata["codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}}], directory, "column must be raw little-endian")
    length = int(metadata["shape"][0])
    _require(metadata["chunk_grid"]["configuration"]["chunk_shape"] == [length], directory, "column must be one chunk")
    dtype = _NUMPY_DATA_TYPES[metadata["data_type"]].newbyteorder("<")
    values = np.memmap(directory / "c" / "0", dtype=dtype, mode="r+" if writable else "r", shape=(length,))
    return values, dict(metadata.get("attributes", {}))


# ---------------------------------------------------------------------------
# Store: manifest, variant table, statistics sidecar and code reads
# ---------------------------------------------------------------------------


def write_manifest(
    root: Path,
    *,
    chromosomes: Sequence[str],
    record_counts: Sequence[int],
    half_sample_counts: Sequence[int],
    attributes: Mapping[str, Any] | None = None,
) -> None:
    if len(chromosomes) != len(record_counts) or not chromosomes or not half_sample_counts:
        raise ValueError("a store needs matching chromosome/record lists and at least one half.")
    if min(record_counts) < 1 or min(half_sample_counts) < 1:
        raise ValueError("every chromosome needs records and every half needs samples.")
    manifest = {
        "format": STORE_FORMAT,
        "chromosomes": list(chromosomes),
        "record_counts": [int(count) for count in record_counts],
        "half_sample_counts": [int(count) for count in half_sample_counts],
        "attributes": dict(attributes or {}),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1))


def read_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / MANIFEST_FILE).read_text())
    if manifest.get("format") != STORE_FORMAT:
        raise ValueError(f"{root} is not an {STORE_FORMAT} store (format={manifest.get('format')!r}).")
    return manifest


def dosage_array_directory(root: Path, half_index: int, chromosome: str) -> Path:
    return root / "dosage" / f"half{half_index}" / chromosome


def variant_column_directory(root: Path, chromosome: str, column: str) -> Path:
    return root / "variants" / chromosome / column


def statistic_column_directory(root: Path, half_index: int, chromosome: str, column: str) -> Path:
    return root / "stats" / f"half{half_index}" / chromosome / column


def write_variant_ids(root: Path, chromosome: str, variant_ids: Sequence[str]) -> None:
    encoded = [variant_id.encode() for variant_id in variant_ids]
    offsets = np.zeros(len(encoded) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum([len(variant_id) for variant_id in encoded], dtype=np.uint64)
    write_column(variant_column_directory(root, chromosome, _ID_BYTES_COLUMN), np.frombuffer(b"".join(encoded), dtype=np.uint8))
    write_column(variant_column_directory(root, chromosome, _ID_OFFSETS_COLUMN), offsets)


@dataclass(frozen=True)
class VariantTable:
    """Columnar per-record metadata, concatenated over chromosomes in store order.

    ``columns`` holds every sidecar column (at least ``REQUIRED_VARIANT_COLUMNS``); categorical
    columns carry their category names in ``legends``.  Variant IDs stay as one byte buffer plus
    offsets and are decoded only for the indices asked for.
    """

    chromosomes: tuple[str, ...]
    chromosome_starts: I64Array
    columns: Mapping[str, NDArray]
    legends: Mapping[str, tuple[str, ...]]
    id_bytes: U8Array
    id_offsets: I64Array

    @property
    def variant_count(self) -> int:
        return int(self.chromosome_starts[-1])

    def chromosome_indices(self, variant_indices: NDArray) -> I64Array:
        return np.searchsorted(self.chromosome_starts, np.asarray(variant_indices), side="right").astype(np.int64) - 1

    def variant_ids(self, variant_indices: NDArray) -> list[str]:
        buffer = self.id_bytes.tobytes()
        return [
            buffer[int(self.id_offsets[index]) : int(self.id_offsets[index + 1])].decode()
            for index in np.asarray(variant_indices, dtype=np.int64)
        ]

    @classmethod
    def read(cls, root: Path, chromosomes: Sequence[str], record_counts: Sequence[int]) -> VariantTable:
        column_names: list[str] | None = None
        per_column: dict[str, list[NDArray]] = {}
        legends: dict[str, tuple[str, ...]] = {}
        id_bytes: list[NDArray] = []
        id_offsets: list[NDArray] = []
        byte_cursor = 0
        for chromosome, record_count in zip(chromosomes, record_counts):
            names = sorted(
                path.name
                for path in (root / "variants" / chromosome).iterdir()
                if path.name not in (_ID_BYTES_COLUMN, _ID_OFFSETS_COLUMN)
            )
            if column_names is None:
                missing = sorted(set(REQUIRED_VARIANT_COLUMNS) - set(names))
                if missing:
                    raise ValueError(f"variant table for {chromosome} lacks required columns {missing}.")
                column_names = names
            elif names != column_names:
                raise ValueError(f"variant columns of {chromosome} differ from those of {chromosomes[0]}.")
            for name in names:
                values, attributes = open_column(variant_column_directory(root, chromosome, name))
                if values.shape[0] != record_count:
                    raise ValueError(f"variant column {chromosome}/{name} has {values.shape[0]} rows, not {record_count}.")
                per_column.setdefault(name, []).append(values)
                if "legend" in attributes:
                    legend = tuple(attributes["legend"])
                    if legends.setdefault(name, legend) != legend:
                        raise ValueError(f"variant column {name} changes its legend at {chromosome}.")
            chromosome_bytes, _ = open_column(variant_column_directory(root, chromosome, _ID_BYTES_COLUMN))
            chromosome_offsets, _ = open_column(variant_column_directory(root, chromosome, _ID_OFFSETS_COLUMN))
            if chromosome_offsets.shape[0] != record_count + 1 or int(chromosome_offsets[-1]) != chromosome_bytes.shape[0]:
                raise ValueError(f"variant ids of {chromosome} do not match its {record_count} records.")
            id_bytes.append(chromosome_bytes)
            id_offsets.append(chromosome_offsets[:-1].astype(np.int64) + byte_cursor)
            byte_cursor += int(chromosome_bytes.shape[0])
        id_offsets.append(np.array([byte_cursor], dtype=np.int64))
        chromosome_starts = np.zeros(len(record_counts) + 1, dtype=np.int64)
        chromosome_starts[1:] = np.cumsum(record_counts)
        return cls(
            chromosomes=tuple(chromosomes),
            chromosome_starts=chromosome_starts,
            columns={name: np.concatenate(parts) for name, parts in per_column.items()},
            legends=legends,
            id_bytes=np.concatenate(id_bytes),
            id_offsets=np.concatenate(id_offsets),
        )


def _reader_thread_count() -> int:
    return len(os.sched_getaffinity(0))


def _ensure_open_file_capacity(required_descriptors: int) -> None:
    """Raise the soft open-file limit to cover one descriptor and one mapping per shard."""
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit < required_descriptors:
        if hard_limit != resource.RLIM_INFINITY and hard_limit < required_descriptors:
            raise RuntimeError(
                f"the store needs {required_descriptors} open files but the hard limit is {hard_limit}."
            )
        resource.setrlimit(resource.RLIMIT_NOFILE, (required_descriptors, hard_limit))


class DosageStore:
    """An svpgs-store v1 opened for reading.

    The variant axis concatenates the chromosomes in manifest order; the sample axis concatenates
    the selected halves (all halves unless ``half_indices`` names a subset).
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
        self.chromosome_starts = np.zeros(len(self.chromosomes) + 1, dtype=np.int64)
        self.chromosome_starts[1:] = np.cumsum(self.record_counts)
        half_counts = [all_half_counts[half] for half in selected]
        self.half_sample_starts = np.zeros(len(selected) + 1, dtype=np.int64)
        self.half_sample_starts[1:] = np.cumsum(half_counts)
        self._arrays: list[list[CodeArray]] = []
        for half, sample_count in zip(selected, half_counts):
            arrays = [CodeArray(dosage_array_directory(self.root, half, chromosome)) for chromosome in self.chromosomes]
            for array, record_count in zip(arrays, self.record_counts):
                if (array.layout.row_count, array.layout.sample_count) != (record_count, sample_count):
                    raise ValueError(f"{array.directory} does not match the manifest's {record_count} x {sample_count}.")
            self._arrays.append(arrays)
        shard_total = sum(array.layout.shard_count for arrays in self._arrays for array in arrays)
        _ensure_open_file_capacity(2 * shard_total + 256)
        self.variants = VariantTable.read(self.root, self.chromosomes, self.record_counts)
        self._pool = ThreadPoolExecutor(max_workers=_reader_thread_count(), thread_name_prefix="dosage-store")

    @property
    def variant_count(self) -> int:
        return int(self.chromosome_starts[-1])

    @property
    def sample_count(self) -> int:
        return int(self.half_sample_starts[-1])

    def statistic(self, name: str) -> NDArray:
        """Per-record sidecar statistic summed over the selected halves, in store variant order."""
        per_chromosome = []
        for chromosome, record_count in zip(self.chromosomes, self.record_counts):
            total: NDArray | None = None
            for half in self.half_indices:
                values, _ = open_column(statistic_column_directory(self.root, half, chromosome, name))
                if values.shape[0] != record_count:
                    raise ValueError(f"statistic {name} of half{half}/{chromosome} has {values.shape[0]} rows.")
                total = values.astype(np.int64) if total is None else total + values.astype(np.int64)
            per_chromosome.append(total)
        return np.concatenate(per_chromosome)

    def _chromosome_pieces(self, variant_start: int, variant_stop: int) -> Iterator[tuple[int, int, int, int]]:
        """Yield (chromosome_index, local_start, local_stop, out_row) covering a global variant range."""
        if not 0 <= variant_start <= variant_stop <= self.variant_count:
            raise IndexError(f"variants [{variant_start}, {variant_stop}) fall outside [0, {self.variant_count}).")
        cursor = variant_start
        while cursor < variant_stop:
            chromosome = int(np.searchsorted(self.chromosome_starts, cursor, side="right")) - 1
            chromosome_start = int(self.chromosome_starts[chromosome])
            piece_stop = min(variant_stop, int(self.chromosome_starts[chromosome + 1]))
            yield chromosome, cursor - chromosome_start, piece_stop - chromosome_start, cursor - variant_start
            cursor = piece_stop

    def codes(self, variant_start: int, variant_stop: int, out: U8Array) -> U8Array:
        """Codes [variants, samples] for a variant range.

        The result is a zero-copy view of the shard's page cache when the store has one half and
        the range sits in one shard; otherwise ``out`` (at least that many rows) is filled and its
        leading rows are returned.  Both hold identical bytes.
        """
        if len(self._arrays) == 1:
            pieces = list(self._chromosome_pieces(variant_start, variant_stop))
            if len(pieces) == 1:
                chromosome, local_start, local_stop, _ = pieces[0]
                array = self._arrays[0][chromosome]
                if len(list(array.shard_pieces(local_start, local_stop))) == 1:
                    return array.view_rows(local_start, local_stop)
        return self.read_codes_into(variant_start, variant_stop, out)

    def advise_codes(self, variant_start: int, variant_stop: int) -> None:
        """Start asynchronous page-cache read-ahead of a variant range in every selected half."""
        for chromosome, local_start, local_stop, _ in self._chromosome_pieces(variant_start, variant_stop):
            for arrays in self._arrays:
                arrays[chromosome].advise_rows(local_start, local_stop)

    def iter_code_views(
        self,
        variant_ranges: Sequence[tuple[int, int]],
        spare: U8Array,
    ) -> Iterator[tuple[int, int, U8Array]]:
        """Yield :meth:`codes` for each range, advising the kernel to read the next range ahead.

        This is the large-RAM path: blocks are page-cache views wherever :meth:`codes` can give one,
        and ``spare`` receives only the ranges that cannot be viewed.
        """
        for position, (start, stop) in enumerate(variant_ranges):
            if position + 1 < len(variant_ranges):
                self.advise_codes(*variant_ranges[position + 1])
            yield start, stop, self.codes(start, stop, spare)

    def read_codes_into(self, variant_start: int, variant_stop: int, out: U8Array) -> U8Array:
        """Fill ``out[:rows]`` with the codes of a variant range using parallel ``preadv`` reads."""
        rows = variant_stop - variant_start
        if out.dtype != np.uint8 or out.ndim != 2 or out.shape[0] < rows or out.shape[1] != self.sample_count:
            raise ValueError(f"out must be uint8 [>= {rows}, {self.sample_count}].")
        target = out[:rows]
        tasks = []
        for chromosome, local_start, local_stop, out_row in self._chromosome_pieces(variant_start, variant_stop):
            for half_position, arrays in enumerate(self._arrays):
                column_start = int(self.half_sample_starts[half_position])
                column_stop = int(self.half_sample_starts[half_position + 1])
                array = arrays[chromosome]
                step = max(array.layout.inner_rows, -(-(local_stop - local_start) // _reader_thread_count()))
                for piece_start in range(local_start, local_stop, step):
                    piece_stop = min(local_stop, piece_start + step)
                    destination = target[
                        out_row + piece_start - local_start : out_row + piece_stop - local_start,
                        column_start:column_stop,
                    ]
                    tasks.append(self._pool.submit(array.read_rows_into, piece_start, piece_stop, destination))
        for task in tasks:
            task.result()
        return target

    def iter_code_blocks(
        self,
        variant_ranges: Sequence[tuple[int, int]],
        buffers: Sequence[U8Array],
    ) -> Iterator[tuple[int, int, U8Array]]:
        """Stream variant ranges through a ring of caller buffers, reading ahead in the background.

        With k buffers, up to k - 1 ranges are read while the caller works on the current one.  A
        yielded block stays valid until the caller asks for the next block.
        """
        if len(buffers) < 2:
            raise ValueError("read-ahead needs at least two buffers.")
        prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dosage-store-prefetch")
        in_flight: deque[Future[U8Array]] = deque()
        try:
            for position, (start, stop) in enumerate(variant_ranges[: len(buffers) - 1]):
                in_flight.append(prefetch.submit(self.read_codes_into, start, stop, buffers[position]))
            for position, (start, stop) in enumerate(variant_ranges):
                block = in_flight.popleft().result()
                yield start, stop, block
                upcoming = position + len(buffers) - 1
                if upcoming < len(variant_ranges):
                    next_start, next_stop = variant_ranges[upcoming]
                    in_flight.append(
                        prefetch.submit(self.read_codes_into, next_start, next_stop, buffers[upcoming % len(buffers)])
                    )
        finally:
            for pending in in_flight:
                pending.cancel()
            prefetch.shutdown(wait=True)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for arrays in self._arrays:
            for array in arrays:
                array.close()

    def __enter__(self) -> DosageStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Exact training-row moments and the standardized design they define
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignedCodeMoments:
    """Exact per-variant sums of s and s**2 (s = code - 127) over a row set."""

    row_count: int
    signed_sums: I64Array
    signed_square_sums: I64Array

    @property
    def scaled_variances(self) -> I64Array:
        """row_count**2 times the population variance of s, an exact integer."""
        return self.row_count * self.signed_square_sums - self.signed_sums * self.signed_sums

    @property
    def means(self) -> F64Array:
        return self.signed_sums / self.row_count

    @property
    def scales(self) -> F64Array:
        return np.sqrt(self.scaled_variances.astype(np.float64)) / self.row_count

    @property
    def dosage_means(self) -> F64Array:
        return (self.means + SIGNED_CODE_OFFSET) / CODES_PER_DOSAGE

    @property
    def dosage_scales(self) -> F64Array:
        return self.scales / CODES_PER_DOSAGE


def _code_sums(codes: U8Array) -> tuple[I64Array, I64Array]:
    """Exact per-row sums of code and code**2 (code**2 <= 64516 fits uint16)."""
    squares = codes.astype(np.uint16)
    np.multiply(squares, squares, out=squares)
    return np.add.reduce(codes, axis=1, dtype=np.int64), np.add.reduce(squares, axis=1, dtype=np.int64)


def _validated_rows(rows: NDArray, sample_count: int) -> I64Array:
    values = np.asarray(rows, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any(np.diff(values) <= 0) or values[0] < 0 or values[-1] >= sample_count:
        raise ValueError("rows must be sorted, distinct sample indices of the store.")
    return values


def signed_code_moments(
    store: DosageStore,
    variant_start: int,
    variant_stop: int,
    training_rows: I64Array,
    *,
    block_rows: int,
) -> SignedCodeMoments:
    """Exact moments of s over ``training_rows`` (sorted, distinct sample indices) in one pass.

    The same pass sums every sample's codes and compares them with the sidecar's ``sum_code`` and
    ``sum_code2``; any difference means the dosage bytes do not match the converter's record and
    raises.
    """
    rows = _validated_rows(training_rows, store.sample_count)
    use_complement = rows.size > store.sample_count // 2
    gathered = np.setdiff1d(np.arange(store.sample_count), rows) if use_complement else rows
    expected_sums = store.statistic("sum_code")[variant_start:variant_stop]
    expected_squares = store.statistic("sum_code2")[variant_start:variant_stop]
    variant_count = variant_stop - variant_start
    code_sums = np.empty(variant_count, dtype=np.int64)
    square_sums = np.empty(variant_count, dtype=np.int64)
    ranges = [(start, min(variant_stop, start + block_rows)) for start in range(variant_start, variant_stop, block_rows)]
    buffers = [np.empty((block_rows, store.sample_count), dtype=np.uint8) for _ in range(2)]
    for start, stop, block in store.iter_code_blocks(ranges, buffers):
        local = slice(start - variant_start, stop - variant_start)
        all_sums, all_squares = _code_sums(block)
        mismatched = np.flatnonzero((all_sums != expected_sums[local]) | (all_squares != expected_squares[local]))
        if mismatched.size:
            raise ValueError(
                f"{mismatched.size} variants starting at {start + int(mismatched[0])} disagree with the sidecar code sums."
            )
        gathered_sums, gathered_squares = _code_sums(block[:, gathered])
        code_sums[local] = all_sums - gathered_sums if use_complement else gathered_sums
        square_sums[local] = all_squares - gathered_squares if use_complement else gathered_squares
    row_count = int(rows.size)
    return SignedCodeMoments(
        row_count=row_count,
        signed_sums=code_sums - SIGNED_CODE_OFFSET * row_count,
        signed_square_sums=square_sums - 2 * SIGNED_CODE_OFFSET * code_sums + _MAXIMUM_SIGNED_PRODUCT * row_count,
    )


def exact_signed_gram(signed: NDArray) -> I64Array:
    """Exact S S^T for signed codes S [variants, rows] via fp32 GEMMs over ``FLOAT32_EXACT_ROWS`` rows.

    Each chunk's partial sums are integers of magnitude <= FLOAT32_EXACT_ROWS * 127**2 < 2**24, so
    every fp32 product-sum is exact whatever order BLAS adds in; chunks accumulate in float64,
    exact below 2**53.
    """
    variant_count, row_count = signed.shape
    total = np.zeros((variant_count, variant_count), dtype=np.float64)
    chunk = np.empty((variant_count, FLOAT32_EXACT_ROWS), dtype=np.float32)
    for row_start in range(0, row_count, FLOAT32_EXACT_ROWS):
        width = min(FLOAT32_EXACT_ROWS, row_count - row_start)
        view = chunk[:, :width]
        view[...] = signed[:, row_start : row_start + width]
        total += view @ view.T
    return total.astype(np.int64)


class QuantizedDosageMatrix:
    """Standardized design over a store: rows are samples, columns a contiguous variant range.

    Column j is x_j = (s_j - mu_j) / sigma_j with training-row moments, evaluated on ``rows``
    (the training rows themselves, or held-out rows scored with training moments).  Every
    operation folds the standardization into integer-valued products of the signed codes.
    """

    def __init__(
        self,
        store: DosageStore,
        variant_start: int,
        variant_stop: int,
        rows: I64Array,
        moments: SignedCodeMoments,
    ) -> None:
        if moments.signed_sums.shape[0] != variant_stop - variant_start:
            raise ValueError("moments must cover exactly the matrix's variant range.")
        monomorphic = np.flatnonzero(moments.scaled_variances <= 0)
        if monomorphic.size:
            raise ValueError(
                f"{monomorphic.size} variants (first {variant_start + int(monomorphic[0])}) have zero training "
                "variance; drop them before standardizing."
            )
        self.store = store
        self.variant_start = variant_start
        self.variant_stop = variant_stop
        self.rows = _validated_rows(rows, store.sample_count)
        self._all_rows = self.rows.size == store.sample_count
        self.moments = moments

    @classmethod
    def from_training_rows(
        cls,
        store: DosageStore,
        variant_start: int,
        variant_stop: int,
        training_rows: I64Array,
        *,
        block_rows: int,
    ) -> QuantizedDosageMatrix:
        """The training design: moments and rows are both the training rows."""
        moments = signed_code_moments(store, variant_start, variant_stop, training_rows, block_rows=block_rows)
        return cls(store, variant_start, variant_stop, training_rows, moments)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.rows.size), self.variant_stop - self.variant_start

    def _row_codes(self, codes: U8Array) -> U8Array:
        return codes if self._all_rows else codes[:, self.rows]

    def _local(self, variant_start: int, variant_stop: int) -> slice:
        if not self.variant_start <= variant_start <= variant_stop <= self.variant_stop:
            raise IndexError(f"variants [{variant_start}, {variant_stop}) fall outside the matrix's range.")
        return slice(variant_start - self.variant_start, variant_stop - self.variant_start)

    def _sub_range_codes(self, variant_start: int, variant_stop: int) -> tuple[slice, U8Array]:
        local = self._local(variant_start, variant_stop)
        buffer = np.empty((variant_stop - variant_start, self.store.sample_count), dtype=np.uint8)
        return local, self._row_codes(self.store.codes(variant_start, variant_stop, buffer))

    def _blocks(self, block_rows: int) -> Iterator[tuple[slice, U8Array]]:
        ranges = [
            (start, min(self.variant_stop, start + block_rows))
            for start in range(self.variant_start, self.variant_stop, block_rows)
        ]
        spare = np.empty((block_rows, self.store.sample_count), dtype=np.uint8)
        for start, stop, block in self.store.iter_code_views(ranges, spare):
            yield self._local(start, stop), self._row_codes(block)

    def standardized_block(self, variant_start: int, variant_stop: int) -> F64Array:
        """Dense x [variants, rows] for a sub-range (tests and small blocks)."""
        local, codes = self._sub_range_codes(variant_start, variant_stop)
        signed = signed_codes(codes, np.empty(codes.shape, dtype=np.float64))
        return (signed - self.moments.means[local, None]) / self.moments.scales[local, None]

    def matvec(self, coefficients: F64Array, *, block_rows: int) -> F64Array:
        """X beta = sum_j s_j (beta_j / sigma_j) - sum_j mu_j beta_j / sigma_j."""
        scaled = np.asarray(coefficients, dtype=np.float64) / self.moments.scales
        result = np.zeros(self.shape[0], dtype=np.float64)
        for local, codes in self._blocks(block_rows):
            signed = signed_codes(codes, np.empty(codes.shape, dtype=np.float64))
            result += scaled[local] @ signed
        return result - float(self.moments.means @ scaled)

    def transpose_matvec(self, vector: F64Array, *, block_rows: int) -> F64Array:
        """X^T v = (S^T v - mu * sum(v)) / sigma."""
        weights = np.asarray(vector, dtype=np.float64)
        signed_products = np.empty(self.shape[1], dtype=np.float64)
        for local, codes in self._blocks(block_rows):
            signed = signed_codes(codes, np.empty(codes.shape, dtype=np.float64))
            signed_products[local] = signed @ weights
        return (signed_products - self.moments.means * float(weights.sum())) / self.moments.scales

    def gram(self, variant_start: int, variant_stop: int) -> F64Array:
        """X_b^T X_b over the matrix rows for a variant sub-range, from the exact signed Gram."""
        local, codes = self._sub_range_codes(variant_start, variant_stop)
        signed_gram = exact_signed_gram(signed_codes(codes, np.empty(codes.shape, dtype=np.int16))).astype(np.float64)
        row_sums = np.add.reduce(codes, axis=1, dtype=np.int64) - SIGNED_CODE_OFFSET * codes.shape[1]
        means = self.moments.means[local]
        centered = (
            signed_gram
            - np.outer(means, row_sums)
            - np.outer(row_sums, means)
            + codes.shape[1] * np.outer(means, means)
        )
        scales = self.moments.scales[local]
        return centered / np.outer(scales, scales)

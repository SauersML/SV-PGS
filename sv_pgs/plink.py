from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sv_pgs._typing import I8Array, NDArray, U8Array

PLINK1_MAGIC = b"\x6c\x1b\x01"
PLINK1_HEADER_SIZE = 3
PLINK_MISSING_INT8 = np.int8(-127)

_DECODE_LOOKUP_A1 = np.array([2, PLINK_MISSING_INT8, 1, 0], dtype=np.int8)
_DECODE_LOOKUP_A2 = np.array([0, PLINK_MISSING_INT8, 1, 2], dtype=np.int8)
_ENCODE_LOOKUP_A1 = np.array([0b11, 0b10, 0b00], dtype=np.uint8)


def _build_byte_decode_lut(per_code_lookup: I8Array) -> I8Array:
    """Precompute a (256, 4) byte → 4-int8 unpacking table.

    Each .bed byte packs 4 samples × 2 bits, low-bit-first. `lut[byte]`
    returns the 4 decoded int8 values in sample order. Single numpy
    advanced-indexing op `lut[packed]` then unpacks the entire batch in
    one shot — one allocation, one indexing kernel — replacing the
    earlier 4-bit-shift + advanced-index-on-2.88GB-codes-array approach
    that allocated several large intermediates and broke up the work
    across many numpy ops (each with its own kernel-call overhead).
    """
    lut = np.empty((256, 4), dtype=np.int8)
    for byte_value in range(256):
        for sample_offset in range(4):
            two_bit_code = (byte_value >> (2 * sample_offset)) & 0b11
            lut[byte_value, sample_offset] = per_code_lookup[two_bit_code]
    return lut


_BYTE_DECODE_LUT_A1 = _build_byte_decode_lut(_DECODE_LOOKUP_A1)
_BYTE_DECODE_LUT_A2 = _build_byte_decode_lut(_DECODE_LOOKUP_A2)

# Large contiguous sample-window reads are latency-bound if we issue one small
# pread per variant. AoU-scale batches are faster when read as large contiguous
# BED spans and sliced in memory, even though that reads extra bytes.
_SAMPLE_WINDOW_STRIPED_MIN_VARIANTS = 512
_SAMPLE_WINDOW_STRIPED_MIN_SPAN_BYTES = 64 * 1024 * 1024
_SAMPLE_WINDOW_STRIPED_TARGET_CHUNK_BYTES = 256 * 1024 * 1024
_SAMPLE_WINDOW_STRIPED_MAX_OVERREAD_RATIO = 32.0

# Sparse-but-sorted variant gathers (for example active post-screening SNPs)
# otherwise issue one pread per short contiguous run. At AoU scale that becomes
# seek latency dominated. Coalesce nearby records into bounded sequential spans,
# then scatter only the requested records into the payload passed to the decoder.
_INDEXED_VARIANT_COALESCE_MIN_VARIANTS = 64
_INDEXED_VARIANT_COALESCE_TARGET_CHUNK_BYTES = 256 * 1024 * 1024
_INDEXED_VARIANT_COALESCE_MAX_OVERREAD_RATIO = 4.0


def _use_coalesced_indexed_variant_reads(indices: NDArray) -> bool:
    """Return True when a variant gather can benefit from sequential spans."""
    resolved = np.asarray(indices, dtype=np.int64)
    if resolved.ndim != 1 or resolved.size < _INDEXED_VARIANT_COALESCE_MIN_VARIANTS:
        return False
    deltas = np.diff(resolved)
    if not np.all(deltas > 0):
        return False
    span_variant_count = int(resolved[-1] - resolved[0] + 1)
    if span_variant_count == int(resolved.size):
        return False
    return span_variant_count <= int(
        np.ceil(float(resolved.size) * _INDEXED_VARIANT_COALESCE_MAX_OVERREAD_RATIO)
    )


def to_bed(
    bed_path: str | Path,
    values: NDArray,
    properties: dict[str, list[Any]],
) -> None:
    resolved_path = Path(bed_path)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("BED matrix must be 2D.")

    sample_count, variant_count = matrix.shape
    bytes_per_variant = _bytes_per_variant(sample_count)

    with resolved_path.open("wb") as bed_handle:
        bed_handle.write(PLINK1_MAGIC)
        for variant_index in range(variant_count):
            packed = _encode_variant(matrix[:, variant_index], bytes_per_variant=bytes_per_variant)
            bed_handle.write(packed)

    sample_ids = [str(sample_id) for sample_id in properties.get("iid", [])]
    family_ids = [str(family_id) for family_id in properties.get("fid", sample_ids)]
    if len(sample_ids) != sample_count or len(family_ids) != sample_count:
        raise ValueError("Sample metadata must match BED row count.")
    fam_path = resolved_path.with_suffix(".fam")
    fam_path.write_text(
        "".join(
            f"{family_id} {sample_id} 0 0 0 -9\n"
            for family_id, sample_id in zip(family_ids, sample_ids, strict=True)
        ),
        encoding="utf-8",
    )

    variant_ids = [str(variant_id) for variant_id in properties.get("sid", [])]
    chromosomes = [str(chromosome) for chromosome in properties.get("chromosome", ["1"] * variant_count)]
    positions = [int(position) for position in properties.get("bp_position", range(1, variant_count + 1))]
    allele_one = [str(value) for value in properties.get("allele_1", ["A"] * variant_count)]
    allele_two = [str(value) for value in properties.get("allele_2", ["C"] * variant_count)]
    if not (
        len(variant_ids)
        == len(chromosomes)
        == len(positions)
        == len(allele_one)
        == len(allele_two)
        == variant_count
    ):
        raise ValueError("Variant metadata must match BED column count.")
    bim_path = resolved_path.with_suffix(".bim")
    bim_path.write_text(
        "".join(
            f"{chromosome} {variant_id} 0 {position} {allele_left} {allele_right}\n"
            for chromosome, variant_id, position, allele_left, allele_right in zip(
                chromosomes,
                variant_ids,
                positions,
                allele_one,
                allele_two,
                strict=True,
            )
        ),
        encoding="utf-8",
    )


@dataclass(slots=True)
class open_bed:
    path: str | Path
    iid_count: int
    sid_count: int
    properties: dict[str, Any] | None = None
    count_A1: bool = True
    num_threads: int | None = None
    _bed_fd: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.iid_count < 1 or self.sid_count < 1:
            raise ValueError("iid_count and sid_count must be positive.")
        with Path(self.path).open("rb") as bed_handle:
            header = bed_handle.read(PLINK1_HEADER_SIZE)
        if header != PLINK1_MAGIC:
            raise ValueError(
                "Invalid PLINK 1 .bed header: expected "
                + PLINK1_MAGIC.hex(" ")
                + ", got "
                + header.hex(" ")
            )
        expected_size = PLINK1_HEADER_SIZE + self.sid_count * _bytes_per_variant(self.iid_count)
        actual_size = Path(self.path).stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "PLINK 1 .bed size does not match iid_count/sid_count: "
                + f"expected {expected_size} bytes, found {actual_size}"
            )
        # Eager open so prefetch threads race-free; lifetime matches reader.
        self._bed_fd = os.open(str(self.path), os.O_RDONLY)
        fadvise = getattr(os, "posix_fadvise", None)
        advice = getattr(os, "POSIX_FADV_RANDOM", None)
        if fadvise is not None and advice is not None:
            try:
                fadvise(self._bed_fd, 0, 0, advice)
            except OSError:
                pass

    def _ensure_fd(self) -> int:
        """Open the .bed file once and reuse the descriptor across threads.

        We deliberately do NOT mmap. mmap can turn storage I/O faults into
        SIGBUS, which terminates the process instead of raising a recoverable
        OSError. We hit exactly this under sustained multi-threaded prefetch:
        a worker thread would page-fault into a previously-evicted region and
        the kernel would raise SIGBUS instead of returning EIO.

        Positional reads are thread-safe (no shared file-offset state) and
        surface I/O errors as OSError, which the prefetch executor can retry
        or propagate cleanly.
        """
        if self._bed_fd is None:
            raise RuntimeError("PLINK reader file descriptor is closed.")
        return self._bed_fd

    def _pread_payload(self, offset: int, length: int) -> U8Array:
        """Positional read of `length` bytes at `offset` into a fresh uint8 array.

        Loops over short reads because filesystems can return fewer bytes than
        requested even mid-file. Returns a numpy-owned buffer with no shared
        mmap pages and no SIGBUS exposure.
        """
        if length <= 0:
            return np.empty((0,), dtype=np.uint8)
        fd = self._ensure_fd()
        buffer = np.empty((length,), dtype=np.uint8)
        view = memoryview(buffer).cast("B")
        bytes_read = 0
        while bytes_read < length:
            chunk_len = os.preadv(fd, [view[bytes_read:]], offset + bytes_read)
            if chunk_len == 0:
                raise OSError(
                    f"Unexpected EOF reading PLINK .bed at offset {offset + bytes_read}: "
                    f"requested {length - bytes_read} bytes, got 0"
                )
            bytes_read += chunk_len
        return buffer

    def _pread_sample_window_payload(
        self,
        *,
        variant_start: int,
        variant_count: int,
        bytes_per_variant: int,
        byte_start: int,
        byte_count: int,
    ) -> U8Array:
        """Read only a contiguous sample byte window from each variant record."""
        if variant_count <= 0 or byte_count <= 0:
            return np.empty((0,), dtype=np.uint8)
        if byte_start < 0 or byte_count < 0 or byte_start + byte_count > bytes_per_variant:
            raise ValueError("PLINK sample byte window is out of bounds.")
        fd = self._ensure_fd()
        payload = np.empty((variant_count * byte_count,), dtype=np.uint8)
        output = memoryview(payload).cast("B")
        for local_variant_index in range(variant_count):
            input_offset = (
                PLINK1_HEADER_SIZE
                + (variant_start + local_variant_index) * bytes_per_variant
                + byte_start
            )
            output_offset = local_variant_index * byte_count
            bytes_read = 0
            while bytes_read < byte_count:
                chunk = os.pread(fd, byte_count - bytes_read, input_offset + bytes_read)
                if not chunk:
                    raise OSError(
                        f"Unexpected EOF reading PLINK .bed sample window at offset "
                        f"{input_offset + bytes_read}: requested {byte_count - bytes_read} bytes, got 0"
                    )
                chunk_len = len(chunk)
                output[output_offset + bytes_read : output_offset + bytes_read + chunk_len] = chunk
                bytes_read += chunk_len
        return payload

    def _pread_indexed_variant_payload(
        self,
        variant_indices: NDArray,
        *,
        bytes_per_variant: int,
    ) -> U8Array:
        """Read arbitrary variant records into one contiguous payload buffer."""
        indices = np.asarray(variant_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("PLINK variant indices must be 1D.")
        if indices.size == 0:
            return np.empty((0,), dtype=np.uint8)
        if bytes_per_variant <= 0:
            raise ValueError("bytes_per_variant must be positive.")

        fd = self._ensure_fd()
        payload = np.empty((int(indices.size) * bytes_per_variant,), dtype=np.uint8)
        output = memoryview(payload).cast("B")
        if _use_coalesced_indexed_variant_reads(indices):
            cursor = 0
            while cursor < int(indices.size):
                first_variant = int(indices[cursor])
                chunk_end = cursor + 1
                last_variant = first_variant
                while chunk_end < int(indices.size):
                    candidate_last = int(indices[chunk_end])
                    span_variant_count = candidate_last - first_variant + 1
                    selected_variant_count = chunk_end - cursor + 1
                    span_bytes = span_variant_count * bytes_per_variant
                    if (
                        span_bytes > _INDEXED_VARIANT_COALESCE_TARGET_CHUNK_BYTES
                        and selected_variant_count > 1
                    ):
                        break
                    if (
                        span_variant_count
                        > selected_variant_count * _INDEXED_VARIANT_COALESCE_MAX_OVERREAD_RATIO
                    ):
                        break
                    last_variant = candidate_last
                    chunk_end += 1

                selected_variant_count = chunk_end - cursor
                span_variant_count = last_variant - first_variant + 1
                output_offset = cursor * bytes_per_variant
                if span_variant_count == selected_variant_count:
                    input_offset = PLINK1_HEADER_SIZE + first_variant * bytes_per_variant
                    byte_count = selected_variant_count * bytes_per_variant
                    bytes_read = 0
                    while bytes_read < byte_count:
                        chunk_len = os.preadv(
                            fd,
                            [output[output_offset + bytes_read : output_offset + byte_count]],
                            input_offset + bytes_read,
                        )
                        if chunk_len == 0:
                            raise OSError(
                                f"Unexpected EOF reading PLINK .bed at offset {input_offset + bytes_read}: "
                                f"requested {byte_count - bytes_read} bytes, got 0"
                            )
                        bytes_read += chunk_len
                else:
                    input_offset = PLINK1_HEADER_SIZE + first_variant * bytes_per_variant
                    span = self._pread_payload(input_offset, span_variant_count * bytes_per_variant)
                    selected_variants = indices[cursor:chunk_end]
                    for output_index, variant_index in enumerate(selected_variants):
                        relative_variant = int(variant_index) - first_variant
                        source_start = relative_variant * bytes_per_variant
                        source_stop = source_start + bytes_per_variant
                        destination_start = output_offset + output_index * bytes_per_variant
                        output[
                            destination_start : destination_start + bytes_per_variant
                        ] = span[source_start:source_stop]
                cursor = chunk_end
            return payload

        cursor = 0
        while cursor < int(indices.size):
            run_start = cursor
            first_variant = int(indices[cursor])
            run_length = 1
            while (
                cursor + run_length < int(indices.size)
                and int(indices[cursor + run_length]) == first_variant + run_length
            ):
                run_length += 1

            input_offset = PLINK1_HEADER_SIZE + first_variant * bytes_per_variant
            output_offset = run_start * bytes_per_variant
            byte_count = run_length * bytes_per_variant
            bytes_read = 0
            while bytes_read < byte_count:
                chunk_len = os.preadv(
                    fd,
                    [output[output_offset + bytes_read : output_offset + byte_count]],
                    input_offset + bytes_read,
                )
                if chunk_len == 0:
                    raise OSError(
                        f"Unexpected EOF reading PLINK .bed at offset {input_offset + bytes_read}: "
                        f"requested {byte_count - bytes_read} bytes, got 0"
                    )
                bytes_read += chunk_len
            cursor += run_length
        return payload

    def _read_sample_window_striped(
        self,
        *,
        variant_start: int,
        variant_count: int,
        bytes_per_variant: int,
        byte_start: int,
        byte_count: int,
        sample_count: int,
        leading_sample_offset: int,
    ) -> I8Array:
        result = np.empty((sample_count, variant_count), dtype=np.int8, order="F")
        variants_per_chunk = max(1, _SAMPLE_WINDOW_STRIPED_TARGET_CHUNK_BYTES // bytes_per_variant)
        for chunk_start in range(0, variant_count, variants_per_chunk):
            chunk_variant_count = min(variants_per_chunk, variant_count - chunk_start)
            input_offset = PLINK1_HEADER_SIZE + (variant_start + chunk_start) * bytes_per_variant
            span = self._pread_payload(input_offset, chunk_variant_count * bytes_per_variant)
            selected = span.reshape(chunk_variant_count, bytes_per_variant)[
                :, byte_start : byte_start + byte_count
            ]
            result[:, chunk_start : chunk_start + chunk_variant_count] = _decode_sample_window_payload(
                selected,
                variant_count=chunk_variant_count,
                byte_count=byte_count,
                sample_count=sample_count,
                leading_sample_offset=leading_sample_offset,
                count_a1=self.count_A1,
            )
        return result

    def close(self) -> None:
        """Release the .bed file descriptor. Idempotent and safe to call twice.

        Long-running pipelines construct many readers; relying solely on
        ``__del__`` leaks fds until garbage collection runs, which on AoU-scale
        jobs can exhaust the per-process fd limit before GC catches up.
        """
        fd = getattr(self, "_bed_fd", None)
        if fd is not None:
            self._bed_fd = None
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "open_bed":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def read(
        self,
        index: Any = None,
        dtype: str | np.dtype[Any] = "float32",
        order: Literal["F", "C"] = "F",
        num_threads: int | None = None,
    ) -> NDArray:
        sample_index, variant_index = _split_index(index)
        resolved_samples = _normalize_index(sample_index, self.iid_count)
        resolved_variants = _normalize_index(variant_index, self.sid_count)
        raw_i8 = self._read_int8_matrix(
            resolved_variants,
            sample_index=resolved_samples,
            num_threads=_resolve_num_threads(
                self.num_threads if num_threads is None else num_threads
            ),
        )

        resolved_dtype = np.dtype(dtype)
        if resolved_dtype == np.dtype(np.int8):
            result = np.asarray(raw_i8, dtype=np.int8)
        else:
            result = raw_i8.astype(resolved_dtype, copy=False)
            result[raw_i8 == PLINK_MISSING_INT8] = np.nan
        if order == "F":
            return np.asfortranarray(result)
        return np.ascontiguousarray(result)

    def _read_int8_matrix(
        self,
        variant_index: slice | NDArray,
        *,
        sample_index: slice | NDArray | None = None,
        num_threads: int = 1,
    ) -> I8Array:
        bytes_per_variant = _bytes_per_variant(self.iid_count)
        if sample_index is None:
            resolved_sample_index: slice | NDArray = slice(0, self.iid_count, 1)
        else:
            resolved_sample_index = sample_index
        sample_slice = resolved_sample_index if isinstance(resolved_sample_index, slice) else None
        if isinstance(variant_index, slice):
            variant_count = max(variant_index.stop - variant_index.start, 0)
            if variant_count == 0:
                sample_count = (
                    _sample_slice_parameters(sample_slice, self.iid_count).sample_count
                    if sample_slice is not None
                    else int(np.asarray(resolved_sample_index, dtype=np.intp).shape[0])
                )
                return np.empty((sample_count, 0), dtype=np.int8)
            if sample_slice is not None:
                params = _sample_slice_parameters(sample_slice, self.iid_count)
                if params.sample_count == 0:
                    return np.empty((0, variant_count), dtype=np.int8)
                if _use_striped_sample_window_reads(
                    variant_count=variant_count,
                    bytes_per_variant=bytes_per_variant,
                    byte_count=params.byte_count,
                ):
                    return self._read_sample_window_striped(
                        variant_start=variant_index.start,
                        variant_count=variant_count,
                        bytes_per_variant=bytes_per_variant,
                        byte_start=params.byte_start,
                        byte_count=params.byte_count,
                        sample_count=params.sample_count,
                        leading_sample_offset=params.leading_sample_offset,
                    )
                payload = self._pread_sample_window_payload(
                    variant_start=variant_index.start,
                    variant_count=variant_count,
                    bytes_per_variant=bytes_per_variant,
                    byte_start=params.byte_start,
                    byte_count=params.byte_count,
                )
                return _decode_sample_window_payload(
                    payload,
                    variant_count=variant_count,
                    byte_count=params.byte_count,
                    sample_count=params.sample_count,
                    leading_sample_offset=params.leading_sample_offset,
                    count_a1=self.count_A1,
                )
            # pread the batch payload into a fresh, numpy-owned buffer. Avoids
            # mmap SIGBUS failures and the shared-file-offset contention of
            # seek+read under multi-threaded prefetch.
            start_offset = PLINK1_HEADER_SIZE + variant_index.start * bytes_per_variant
            payload = self._pread_payload(start_offset, variant_count * bytes_per_variant)
            if isinstance(resolved_sample_index, np.ndarray):
                return _decode_payload_sample_indices_parallel(
                    payload,
                    iid_count=self.iid_count,
                    variant_count=variant_count,
                    bytes_per_variant=bytes_per_variant,
                    sample_indices=resolved_sample_index,
                    count_a1=self.count_A1,
                    num_threads=num_threads,
                )
            full = _decode_payload_parallel(
                payload,
                iid_count=self.iid_count,
                variant_count=variant_count,
                bytes_per_variant=bytes_per_variant,
                count_a1=self.count_A1,
                num_threads=num_threads,
            )
            return full[resolved_sample_index, :]

        if isinstance(resolved_sample_index, np.ndarray):
            sample_indices = np.asarray(resolved_sample_index, dtype=np.intp)
            if sample_indices.size == 0:
                return np.empty((0, variant_index.shape[0]), dtype=np.int8)
            n_variants = int(variant_index.shape[0])
            payload = self._pread_indexed_variant_payload(
                variant_index,
                bytes_per_variant=bytes_per_variant,
            )
            return _decode_payload_sample_indices_parallel(
                payload,
                iid_count=self.iid_count,
                variant_count=n_variants,
                bytes_per_variant=bytes_per_variant,
                sample_indices=sample_indices,
                count_a1=self.count_A1,
                num_threads=num_threads,
            )

        if sample_slice is not None:
            params = _sample_slice_parameters(sample_slice, self.iid_count)
            if params.sample_count == 0:
                return np.empty((0, variant_index.shape[0]), dtype=np.int8)
            result = np.empty((params.sample_count, variant_index.shape[0]), dtype=np.int8)
            with Path(self.path).open("rb") as bed_handle:
                for output_index, bed_variant_index in enumerate(variant_index):
                    bed_handle.seek(
                        PLINK1_HEADER_SIZE
                        + int(bed_variant_index) * bytes_per_variant
                        + params.byte_start
                    )
                    payload_bytes = bed_handle.read(params.byte_count)
                    result[:, output_index] = _decode_sample_window_payload(
                        payload_bytes,
                        variant_count=1,
                        byte_count=params.byte_count,
                        sample_count=params.sample_count,
                        leading_sample_offset=params.leading_sample_offset,
                        count_a1=self.count_A1,
                    )[:, 0]
            return result

        payload = self._pread_indexed_variant_payload(
            variant_index,
            bytes_per_variant=bytes_per_variant,
        )
        result = _decode_payload_parallel(
            payload,
            iid_count=self.iid_count,
            variant_count=int(variant_index.shape[0]),
            bytes_per_variant=bytes_per_variant,
            count_a1=self.count_A1,
            num_threads=num_threads,
        )
        return result[resolved_sample_index, :]


def _bytes_per_variant(sample_count: int) -> int:
    return (int(sample_count) + 3) // 4


def _resolve_num_threads(num_threads: int | None) -> int:
    if num_threads is None:
        return min(os.cpu_count() or 8, 16)
    resolved = int(num_threads)
    if resolved < 1:
        raise ValueError("num_threads must be at least 1.")
    return resolved


def _variant_chunks(variant_count: int, num_threads: int) -> list[tuple[int, int]]:
    if variant_count <= 0:
        return []
    thread_count = min(num_threads, variant_count)
    chunk_size = (variant_count + thread_count - 1) // thread_count
    return [
        (chunk_start, min(chunk_start + chunk_size, variant_count))
        for chunk_start in range(0, variant_count, chunk_size)
    ]


def _use_striped_sample_window_reads(
    *,
    variant_count: int,
    bytes_per_variant: int,
    byte_count: int,
) -> bool:
    if variant_count < _SAMPLE_WINDOW_STRIPED_MIN_VARIANTS:
        return False
    if bytes_per_variant <= 0 or byte_count <= 0:
        return False
    full_span_bytes = variant_count * bytes_per_variant
    if full_span_bytes < _SAMPLE_WINDOW_STRIPED_MIN_SPAN_BYTES:
        return False
    overread_ratio = bytes_per_variant / byte_count
    return overread_ratio <= _SAMPLE_WINDOW_STRIPED_MAX_OVERREAD_RATIO


def _split_index(index: Any) -> tuple[Any, Any]:
    if index is None:
        return None, None
    if not isinstance(index, tuple) or len(index) != 2:
        raise ValueError("index must be a (sample_index, variant_index) tuple.")
    return index


def _normalize_index(index: Any, limit: int) -> slice | NDArray:
    if index is None:
        return slice(0, limit, 1)
    if isinstance(index, slice):
        start, stop, step = index.indices(limit)
        if step == 1:
            return slice(start, stop, 1)
        return np.arange(start, stop, step, dtype=np.intp)
    if np.isscalar(index):
        value = int(np.asarray(index, dtype=np.intp).item())
        if value < 0:
            value += limit
        if value < 0 or value >= limit:
            raise IndexError("PLINK index out of bounds.")
        return np.asarray([value], dtype=np.intp)

    values = np.asarray(index)
    if values.dtype == np.bool_:
        if values.ndim != 1 or values.shape[0] != limit:
            raise ValueError("Boolean PLINK index mask must be 1D and match the axis length.")
        return np.flatnonzero(values).astype(np.intp, copy=False)
    values = np.asarray(values, dtype=np.intp)
    if values.ndim != 1:
        raise ValueError("PLINK indices must be 1D.")
    normalized = values.copy()
    normalized[normalized < 0] += limit
    if np.any(normalized < 0) or np.any(normalized >= limit):
        raise IndexError("PLINK index out of bounds.")
    return np.ascontiguousarray(normalized, dtype=np.intp)


@dataclass(frozen=True, slots=True)
class _SampleSliceParameters:
    sample_count: int
    byte_start: int
    byte_count: int
    leading_sample_offset: int


def _sample_slice_parameters(sample_slice: slice, iid_count: int) -> _SampleSliceParameters:
    start, stop, step = sample_slice.indices(iid_count)
    if step != 1:
        raise ValueError("PLINK sample slices must have step=1.")
    sample_count = max(stop - start, 0)
    if sample_count == 0:
        return _SampleSliceParameters(
            sample_count=0,
            byte_start=0,
            byte_count=0,
            leading_sample_offset=0,
        )
    byte_start = start // 4
    byte_stop = (stop + 3) // 4
    return _SampleSliceParameters(
        sample_count=sample_count,
        byte_start=byte_start,
        byte_count=byte_stop - byte_start,
        leading_sample_offset=start % 4,
    )


def _decode_sample_window_payload(
    payload: bytes | NDArray,
    *,
    variant_count: int,
    byte_count: int,
    sample_count: int,
    leading_sample_offset: int,
    count_a1: bool,
) -> I8Array:
    if sample_count == 0:
        return np.empty((0, variant_count), dtype=np.int8)
    if byte_count < 1:
        raise ValueError("PLINK sample byte window must contain at least one byte.")
    if leading_sample_offset < 0 or leading_sample_offset > 3:
        raise ValueError("PLINK leading sample offset must be in [0, 3].")
    if isinstance(payload, np.ndarray):
        raw_bytes = np.ascontiguousarray(payload, dtype=np.uint8)
    else:
        raw_bytes = np.frombuffer(payload, dtype=np.uint8)
    if raw_bytes.size != variant_count * byte_count:
        raise ValueError("Unexpected PLINK 1 .bed sample-window payload length.")

    packed = raw_bytes.reshape(variant_count, byte_count)
    lut = _BYTE_DECODE_LUT_A1 if count_a1 else _BYTE_DECODE_LUT_A2
    decoded = lut[packed].reshape(variant_count, byte_count * 4)
    sliced: I8Array = decoded[:, leading_sample_offset : leading_sample_offset + sample_count].T
    return sliced


def _decode_payload_sample_indices(
    payload: bytes | NDArray,
    *,
    iid_count: int,
    variant_count: int,
    bytes_per_variant: int,
    sample_indices: NDArray,
    count_a1: bool,
) -> I8Array:
    samples = np.asarray(sample_indices, dtype=np.intp)
    if samples.ndim != 1:
        raise ValueError("PLINK sample indices must be 1D.")
    if samples.size == 0:
        return np.empty((0, variant_count), dtype=np.int8)
    if np.any(samples < 0) or np.any(samples >= iid_count):
        raise IndexError("PLINK sample index out of bounds.")
    if isinstance(payload, np.ndarray):
        raw_bytes = np.asarray(payload, dtype=np.uint8)
    else:
        raw_bytes = np.frombuffer(payload, dtype=np.uint8)
    if raw_bytes.size != variant_count * bytes_per_variant:
        raise ValueError("Unexpected PLINK 1 .bed payload length.")

    byte_indices = samples // 4
    sample_offsets = samples % 4
    packed_matrix = raw_bytes.reshape(variant_count, bytes_per_variant)
    lut = _BYTE_DECODE_LUT_A1 if count_a1 else _BYTE_DECODE_LUT_A2
    result = np.empty((samples.shape[0], variant_count), dtype=np.int8, order="F")
    for offset in range(4):
        output_positions = np.flatnonzero(sample_offsets == offset)
        if output_positions.size == 0:
            continue
        packed = packed_matrix[:, byte_indices[output_positions]]
        result[output_positions, :] = lut[np.ascontiguousarray(packed, dtype=np.uint8), offset].T
    return result


def _decode_payload_sample_indices_parallel(
    payload: bytes | NDArray,
    *,
    iid_count: int,
    variant_count: int,
    bytes_per_variant: int,
    sample_indices: NDArray,
    count_a1: bool,
    num_threads: int,
) -> I8Array:
    if num_threads <= 1 or variant_count <= 1:
        return _decode_payload_sample_indices(
            payload,
            iid_count=iid_count,
            variant_count=variant_count,
            bytes_per_variant=bytes_per_variant,
            sample_indices=sample_indices,
            count_a1=count_a1,
        )
    samples = np.asarray(sample_indices, dtype=np.intp)
    if samples.ndim != 1:
        raise ValueError("PLINK sample indices must be 1D.")
    if samples.size == 0:
        return np.empty((0, variant_count), dtype=np.int8)
    if np.any(samples < 0) or np.any(samples >= iid_count):
        raise IndexError("PLINK sample index out of bounds.")
    if isinstance(payload, np.ndarray):
        raw_bytes = np.ascontiguousarray(payload, dtype=np.uint8)
    else:
        raw_bytes = np.frombuffer(payload, dtype=np.uint8)
    if raw_bytes.size != variant_count * bytes_per_variant:
        raise ValueError("Unexpected PLINK 1 .bed payload length.")

    result = np.empty((samples.shape[0], variant_count), dtype=np.int8, order="F")
    packed_matrix = raw_bytes.reshape(variant_count, bytes_per_variant)

    def decode_chunk(chunk_start: int, chunk_stop: int) -> None:
        chunk_payload = packed_matrix[chunk_start:chunk_stop, :].reshape(-1)
        result[:, chunk_start:chunk_stop] = _decode_payload_sample_indices(
            chunk_payload,
            iid_count=iid_count,
            variant_count=chunk_stop - chunk_start,
            bytes_per_variant=bytes_per_variant,
            sample_indices=samples,
            count_a1=count_a1,
        )

    with ThreadPoolExecutor(max_workers=min(num_threads, variant_count)) as executor:
        futures = [
            executor.submit(decode_chunk, chunk_start, chunk_stop)
            for chunk_start, chunk_stop in _variant_chunks(variant_count, num_threads)
        ]
        for future in futures:
            future.result()
    return result


def _decode_payload(
    payload: bytes | NDArray,
    *,
    iid_count: int,
    variant_count: int,
    bytes_per_variant: int,
    count_a1: bool,
) -> I8Array:
    """Decode a .bed payload (variant_count × bytes_per_variant packed bytes)
    into an (iid_count, variant_count) int8 array.

    Implementation: single 256-entry byte→4×int8 lookup. The old approach
    allocated a (variant_count, bytes_per_variant*4) uint8 codes table,
    wrote into it 4× via bit-shift + mask + slice-stride, then did a
    second advanced-indexing pass via the 4-entry code-to-int8 LUT — six
    passes over multi-GB intermediates, GIL-released or not, plus an
    optional thread-pool wrapper that on some hardware paid more overhead
    than it saved. One LUT swap collapses all of it into a single numpy
    indexing kernel that's strictly memory-bandwidth-bound.
    """
    if isinstance(payload, np.ndarray):
        raw_bytes = np.ascontiguousarray(payload, dtype=np.uint8)
    else:
        raw_bytes = np.frombuffer(payload, dtype=np.uint8)
    if raw_bytes.size != variant_count * bytes_per_variant:
        raise ValueError("Unexpected PLINK 1 .bed payload length.")

    packed = raw_bytes.reshape(variant_count, bytes_per_variant)
    lut = _BYTE_DECODE_LUT_A1 if count_a1 else _BYTE_DECODE_LUT_A2
    # lut[packed] → (variant_count, bytes_per_variant, 4) int8. One alloc,
    # one indexing op. Reshape collapses the last two axes back into a flat
    # padded-sample axis; the .T returns the F-order view the caller wants.
    decoded = lut[packed]
    result: I8Array = decoded.reshape(variant_count, bytes_per_variant * 4)[:, :iid_count].T
    return result


def _decode_payload_parallel(
    payload: bytes | NDArray,
    *,
    iid_count: int,
    variant_count: int,
    bytes_per_variant: int,
    count_a1: bool,
    num_threads: int,
) -> I8Array:
    if num_threads <= 1 or variant_count <= 1:
        return _decode_payload(
            payload,
            iid_count=iid_count,
            variant_count=variant_count,
            bytes_per_variant=bytes_per_variant,
            count_a1=count_a1,
        )
    if isinstance(payload, np.ndarray):
        raw_bytes = np.ascontiguousarray(payload, dtype=np.uint8)
    else:
        raw_bytes = np.frombuffer(payload, dtype=np.uint8)
    if raw_bytes.size != variant_count * bytes_per_variant:
        raise ValueError("Unexpected PLINK 1 .bed payload length.")

    decoded_matrix = np.empty((variant_count, bytes_per_variant * 4), dtype=np.int8)
    packed_matrix = raw_bytes.reshape(variant_count, bytes_per_variant)
    lut = _BYTE_DECODE_LUT_A1 if count_a1 else _BYTE_DECODE_LUT_A2

    def decode_chunk(chunk_start: int, chunk_stop: int) -> None:
        packed = packed_matrix[chunk_start:chunk_stop, :]
        decoded_matrix[chunk_start:chunk_stop, :] = lut[packed].reshape(
            chunk_stop - chunk_start,
            bytes_per_variant * 4,
        )

    with ThreadPoolExecutor(max_workers=min(num_threads, variant_count)) as executor:
        futures = [
            executor.submit(decode_chunk, chunk_start, chunk_stop)
            for chunk_start, chunk_stop in _variant_chunks(variant_count, num_threads)
        ]
        for future in futures:
            future.result()
    return decoded_matrix[:, :iid_count].T


def _encode_variant(column: NDArray, *, bytes_per_variant: int) -> bytes:
    observed = ~np.isnan(column)
    rounded = np.rint(column[observed])
    if not np.all(np.isclose(column[observed], rounded)):
        raise ValueError("PLINK BED supports only hardcall dosages 0, 1, 2, or NaN.")
    rounded_i8 = rounded.astype(np.int8, copy=False)
    if np.any((rounded_i8 < 0) | (rounded_i8 > 2)):
        raise ValueError("PLINK BED supports only hardcall dosages 0, 1, 2, or NaN.")

    codes = np.zeros(bytes_per_variant * 4, dtype=np.uint8)
    sample_codes = codes[: column.shape[0]]
    sample_codes[~observed] = 0b01
    sample_codes[observed] = _ENCODE_LOOKUP_A1[rounded_i8]
    packed = (
        codes[0::4]
        | (codes[1::4] << 2)
        | (codes[2::4] << 4)
        | (codes[3::4] << 6)
    )
    return packed.tobytes()

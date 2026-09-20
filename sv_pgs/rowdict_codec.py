"""Row-dictionary codec for the dosage store: lossless, every row decodable alone, GPU-decodable.

One inner chunk of R rows of n uint8 codes is encoded as::

    [row frame sizes: R little-endian uint32]
    [R row frames, back to back, in row order]

and the store's chain appends the chunk's crc32c.  A row frame is::

    [u8 depth k]
    [dictionary: the row's 2^k most frequent code values, most frequent first, 2^k bytes]
    [slots: one k-bit dictionary position per sample, bit-packed little-endian, ceil(n k / 8) bytes]
    [exception samples: E little-endian uint16 (n <= 2^16) or uint32]
    [exception codes: E bytes]

with E = (frame size - 1 - 2^k - ceil(n k / 8)) / (sample width + 1), so no field repeats what
the size already says.  A sample whose code is not in the dictionary is an exception: its slot
holds position 0 and its (sample, code) pair overwrites that value once the slots are decoded.
k is the exact minimizer of the frame's bytes over k in 0..8, so no threshold appears: k = 0 is
the sparse case (the row's mode plus its exceptions), k = 8 the dense one (no exceptions).

A slot of k <= 8 bits spans at most two bytes, so a decoder reads a 16-bit window at each
sample's bit; the chunk's crc32c always follows its last frame, so the window never leaves the
chunk.  The GPU decoder writes every output byte once from the slots, then the exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sv_pgs._typing import I64Array, U8Array

BITS_PER_CODE = np.iinfo(np.uint8).bits
CODE_VALUES = 1 << BITS_PER_CODE
MAXIMUM_DEPTH = BITS_PER_CODE
ROW_SIZE_DTYPE = np.dtype("<u4")
_DEPTH_BYTES = np.dtype(np.uint8).itemsize


def exception_sample_dtype(sample_count: int) -> np.dtype[Any]:
    """Stored exception sample index: the narrower of little-endian uint16 / uint32 that holds n - 1."""
    return np.dtype("<u2") if sample_count <= np.iinfo(np.uint16).max + 1 else np.dtype("<u4")


def slot_bytes(sample_count: int, depth: Any) -> Any:
    return -(-(sample_count * depth) // BITS_PER_CODE)


def _histograms(codes: U8Array) -> I64Array:
    rows = codes.shape[0]
    offsets = codes.astype(np.int64) + CODE_VALUES * np.arange(rows, dtype=np.int64)[:, None]
    return np.bincount(offsets.ravel(), minlength=CODE_VALUES * rows).reshape(rows, CODE_VALUES)


def encode_chunk(codes: U8Array) -> bytes:
    """Encode uint8 codes [R, n] as one chunk payload (row size table, then one frame per row)."""
    if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] < 1:
        raise ValueError(f"codes must be uint8 [rows, samples >= 1]; got {codes.dtype} {codes.shape}.")
    rows, samples = codes.shape
    counts = _histograms(codes)
    order = np.argsort(-counts, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(CODE_VALUES), order.shape), axis=1)
    covered = np.cumsum(np.take_along_axis(counts, order, axis=1), axis=1)
    depths = np.arange(MAXIMUM_DEPTH + 1)
    sizes = 1 << depths
    sample_dtype = exception_sample_dtype(samples)
    frame_bytes = (
        _DEPTH_BYTES + sizes[None, :] + slot_bytes(samples, depths)[None, :]
        + (sample_dtype.itemsize + 1) * (samples - covered[:, sizes - 1])
    )
    depth = np.argmin(frame_bytes, axis=1)
    frame_size = frame_bytes[np.arange(rows), depth]
    table = rows * ROW_SIZE_DTYPE.itemsize
    starts = table + np.concatenate(([0], np.cumsum(frame_size)[:-1])).astype(np.int64)
    payload = np.zeros(table + int(frame_size.sum()), dtype=np.uint8)
    payload[:table] = frame_size.astype(ROW_SIZE_DTYPE).view(np.uint8)
    payload[starts] = depth.astype(np.uint8)
    for bits in np.unique(depth).tolist():
        members = np.flatnonzero(depth == bits)
        dictionary_size = 1 << bits
        member_codes = codes[members]
        slot = np.take_along_axis(rank[members], member_codes.astype(np.intp), axis=1)
        outside = slot >= dictionary_size
        slot[outside] = 0
        dictionary_at = starts[members, None] + _DEPTH_BYTES + np.arange(dictionary_size)[None, :]
        payload[dictionary_at] = order[members, :dictionary_size]
        slot_at = starts[members] + _DEPTH_BYTES + dictionary_size
        if bits:
            planes = ((slot[:, :, None] >> np.arange(bits)) & 1).astype(np.uint8)
            packed = np.packbits(planes.reshape(members.shape[0], samples * bits), axis=1, bitorder="little")
            payload[slot_at[:, None] + np.arange(packed.shape[1])[None, :]] = packed
        exception_rows, exception_samples = np.nonzero(outside)
        if exception_rows.shape[0] == 0:
            continue
        exception_counts = np.bincount(exception_rows, minlength=members.shape[0])
        first = np.concatenate(([0], np.cumsum(exception_counts)[:-1]))
        within = np.arange(exception_rows.shape[0]) - first[exception_rows]
        samples_at = slot_at[exception_rows] + int(slot_bytes(samples, bits)) + within * sample_dtype.itemsize
        sample_bytes = exception_samples.astype(sample_dtype).view(np.uint8).reshape(-1, sample_dtype.itemsize)
        payload[samples_at[:, None] + np.arange(sample_dtype.itemsize)[None, :]] = sample_bytes
        codes_at = (
            slot_at[exception_rows] + int(slot_bytes(samples, bits))
            + exception_counts[exception_rows] * sample_dtype.itemsize + within
        )
        payload[codes_at] = member_codes[exception_rows, exception_samples]
    return payload.tobytes()


@dataclass(frozen=True)
class RowFrames:
    """Where each wanted row's frame sits in one byte buffer, with its parsed fields."""

    depth: I64Array
    exception_count: I64Array
    dictionary_offset: I64Array
    slot_offset: I64Array
    exception_offset: I64Array


def _gather_uint(buffer: U8Array, offsets: I64Array, dtype: np.dtype[Any]) -> I64Array:
    """Little-endian unsigned integers of ``dtype`` at arbitrary byte offsets."""
    value = np.zeros(offsets.shape, dtype=np.int64)
    for byte in range(dtype.itemsize):
        value |= buffer[offsets + byte].astype(np.int64) << (BITS_PER_CODE * byte)
    return value


def row_frames(
    buffer: U8Array, chunk_offsets: I64Array, chunk_sizes: I64Array, chunk_rows: int, sample_count: int, rows: I64Array
) -> RowFrames:
    """Locate rows ``rows`` (chunk-major: ``chunk * chunk_rows + row``) of chunks at ``chunk_offsets``.

    ``chunk_sizes`` are the payload sizes (crc32c excluded).  Every chunk's size table and every
    frame header are checked against the sizes; a chunk that does not parse raises.
    """
    table = chunk_rows * ROW_SIZE_DTYPE.itemsize
    entries = chunk_offsets[:, None] + ROW_SIZE_DTYPE.itemsize * np.arange(chunk_rows)[None, :]
    sizes = _gather_uint(buffer, entries.ravel(), ROW_SIZE_DTYPE).reshape(chunk_offsets.shape[0], chunk_rows)
    if not np.array_equal(table + sizes.sum(axis=1), chunk_sizes) or np.any(sizes < _DEPTH_BYTES):
        raise ValueError("a rowdict chunk's row size table does not match its stored size.")
    starts = chunk_offsets[:, None] + table + np.cumsum(sizes, axis=1) - sizes
    offsets, frame_size = starts.ravel()[rows], sizes.ravel()[rows]
    depth = buffer[offsets].astype(np.int64)
    if np.any(depth > MAXIMUM_DEPTH):
        raise ValueError("a rowdict row frame has a depth above 8.")
    dictionary_offset = offsets + _DEPTH_BYTES
    slot_offset = dictionary_offset + (1 << depth)
    exception_offset = slot_offset + slot_bytes(sample_count, depth)
    exception_count, remainder = np.divmod(offsets + frame_size - exception_offset, exception_sample_dtype(sample_count).itemsize + 1)
    if np.any(remainder) or np.any(exception_count < 0):
        raise ValueError("a rowdict row frame's size does not fit its depth.")
    return RowFrames(depth, exception_count, dictionary_offset, slot_offset, exception_offset)


def decode_rows(buffer: U8Array, frames: RowFrames, sample_count: int, out: U8Array) -> None:
    """Decode located row frames into ``out`` [rows, n] on the CPU (the reference decoder)."""
    sample_dtype = exception_sample_dtype(sample_count)
    for bits in np.unique(frames.depth).tolist():
        members = np.flatnonzero(frames.depth == bits)
        dictionary_size = 1 << bits
        dictionaries = buffer[frames.dictionary_offset[members, None] + np.arange(dictionary_size)[None, :]]
        if not bits:
            out[members] = dictionaries
            continue
        width = int(slot_bytes(sample_count, bits))
        packed = np.zeros((members.shape[0], width + 1), dtype=np.uint16)
        packed[:, :width] = buffer[frames.slot_offset[members, None] + np.arange(width)[None, :]]
        byte, shift = np.divmod(np.arange(sample_count, dtype=np.int64) * bits, BITS_PER_CODE)
        window = packed[:, byte] | (packed[:, byte + 1] << np.uint16(BITS_PER_CODE))
        slot = (window >> shift.astype(np.uint16)) & np.uint16(dictionary_size - 1)
        out[members] = np.take_along_axis(dictionaries, slot.astype(np.intp), axis=1)
    counts = frames.exception_count
    exception_rows = np.repeat(np.arange(counts.shape[0]), counts)
    if exception_rows.shape[0] == 0:
        return
    within = np.arange(exception_rows.shape[0]) - (np.cumsum(counts) - counts)[exception_rows]
    samples = _gather_uint(buffer, frames.exception_offset[exception_rows] + within * sample_dtype.itemsize, sample_dtype)
    if np.any(samples >= sample_count):
        raise ValueError("a rowdict exception names a sample outside the row.")
    code_at = frames.exception_offset[exception_rows] + counts[exception_rows] * sample_dtype.itemsize + within
    out[exception_rows, samples] = buffer[code_at]


_GPU_SOURCE = r"""
extern "C" __global__
void decode_row_slots(const unsigned char* __restrict__ frames, const long long* __restrict__ dictionary_offset,
                      const long long* __restrict__ slot_offset, const long long* __restrict__ depth,
                      unsigned char* __restrict__ out, long long first_row, long long samples, long long out_stride) {
    // One block row per variant row: the row's dictionary is staged in shared memory, then each
    // thread decodes samples from a 16-bit window at the sample's first slot bit.
    __shared__ unsigned char values[256];
    long long row = first_row + blockIdx.y;
    int bits = (int)depth[row];
    int size = 1 << bits;
    for (int at = threadIdx.x; at < size; at += blockDim.x) values[at] = frames[dictionary_offset[row] + at];
    __syncthreads();
    const unsigned char* base = frames + slot_offset[row];
    unsigned int mask = (1u << bits) - 1u;
    unsigned char* target = out + row * out_stride;
    for (long long sample = (long long)blockIdx.x * blockDim.x + threadIdx.x; sample < samples; sample += (long long)gridDim.x * blockDim.x) {
        unsigned int slot = 0;
        if (bits) {
            long long bit = sample * bits;
            const unsigned char* at = base + (bit >> 3);
            unsigned int window = (unsigned int)at[0] | ((unsigned int)at[1] << 8);
            slot = (window >> (bit & 7)) & mask;
        }
        target[sample] = values[slot];
    }
}

extern "C" __global__
void scatter_row_exceptions(const unsigned char* __restrict__ frames, const long long* __restrict__ exception_offset,
                            const long long* __restrict__ exception_count, unsigned char* __restrict__ out,
                            long long first_row, long long out_stride, int sample_bytes) {
    // One block per variant row; its threads write the row's exceptions over the slot values.
    long long row = first_row + blockIdx.x;
    long long count = exception_count[row];
    const unsigned char* samples = frames + exception_offset[row];
    const unsigned char* codes = samples + count * sample_bytes;
    unsigned char* target = out + row * out_stride;
    for (long long at = threadIdx.x; at < count; at += blockDim.x) {
        unsigned int sample = 0;
        for (int byte = 0; byte < sample_bytes; ++byte) sample |= (unsigned int)samples[at * sample_bytes + byte] << (8 * byte);
        target[sample] = codes[at];
    }
}
"""


class GpuRowDecoder:
    """Decode located row frames on the GPU, writing each output byte once (slots, then exceptions)."""

    def __init__(self, cupy: Any) -> None:
        self.cupy = cupy
        module = cupy.RawModule(code=_GPU_SOURCE)
        self._slots = module.get_function("decode_row_slots")
        self._exceptions = module.get_function("scatter_row_exceptions")
        attributes = cupy.cuda.Device().attributes
        self._threads = int(attributes["MaxThreadsPerBlock"])
        self._resident_blocks = int(attributes["MultiProcessorCount"]) * (
            int(attributes["MaxThreadsPerMultiProcessor"]) // self._threads
        )
        self._grid_rows = int(attributes["MaxGridDimY"])
        self._grid_blocks = int(attributes["MaxGridDimX"])

    def decode(self, device_buffer: Any, frames: RowFrames, sample_count: int, out: Any, stream: Any = None) -> None:
        """``device_buffer`` holds the chunks the frames lie in; ``out`` is uint8 [rows, n] with contiguous rows.

        The kernels run on ``stream``, or else on the current stream.
        """
        cp = self.cupy
        rows = int(frames.depth.shape[0])
        if rows == 0:
            return
        if out.shape != (rows, sample_count) or out.dtype != cp.uint8 or out.strides[1] != 1:
            raise ValueError(f"out must be uint8 [{rows}, {sample_count}] with contiguous rows.")
        with stream if stream is not None else cp.cuda.get_current_stream():
            # The frames' fields go to the device in one copy.
            fields = cp.asarray(np.stack([
                frames.dictionary_offset, frames.slot_offset, frames.depth, frames.exception_offset, frames.exception_count
            ]))
            dictionary_offset, slot_offset, depth, exception_offset, exception_count = fields
            stride = np.int64(out.strides[0])
            # Enough blocks per row to fill the device when the rows alone do not.
            blocks_per_row = max(1, min(-(-sample_count // self._threads), self._resident_blocks // min(rows, self._resident_blocks)))
            for first in range(0, rows, self._grid_rows):
                self._slots(
                    (blocks_per_row, min(self._grid_rows, rows - first)), (self._threads,),
                    (device_buffer, dictionary_offset, slot_offset, depth, out, np.int64(first), np.int64(sample_count), stride),
                )
            if not int(frames.exception_count.sum()):
                return
            sample_bytes = np.int32(exception_sample_dtype(sample_count).itemsize)
            for first in range(0, rows, self._grid_blocks):
                self._exceptions(
                    (min(self._grid_blocks, rows - first),), (self._threads,),
                    (device_buffer, exception_offset, exception_count, out, np.int64(first), stride, sample_bytes),
                )

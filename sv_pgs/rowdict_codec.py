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


CRC32C_POLYNOMIAL = 0x82F63B78
"""The reflected Castagnoli polynomial: the store chain's crc32c."""


def _crc32c_table() -> np.ndarray:
    table = np.arange(CODE_VALUES, dtype=np.uint32)
    for _ in range(BITS_PER_CODE):
        table = np.where(table & 1, (table >> np.uint32(1)) ^ np.uint32(CRC32C_POLYNOMIAL), table >> np.uint32(1)).astype(np.uint32)
    return table


def _multiply_mod_polynomial(a: int, b: int) -> int:
    """a(x) b(x) mod P(x) in the reflected bit order (zlib's multmodp)."""
    top = 1 << 31
    product = 0
    while True:
        if a & top:
            product ^= b
            if not a & (top - 1):
                return product
        top >>= 1
        b = (b >> 1) ^ CRC32C_POLYNOMIAL if b & 1 else b >> 1


def _power_table() -> list[int]:
    """x^(2^k) mod P for k = 0..31 (zlib's x2n_table)."""
    power = 1 << 30
    table = [power]
    for _ in range(31):
        power = _multiply_mod_polynomial(power, power)
        table.append(power)
    return table


_POWERS = _power_table()


def byte_shift(byte_count: int) -> int:
    """x^(8 byte_count) mod P: crc32c(A + B) = byte_shift(len(B)) (x) crc32c(A) ^ crc32c(B) (zlib's crc32_combine)."""
    power, exponent, bit = 1 << 31, int(byte_count), 3
    while exponent:
        if exponent & 1:
            power = _multiply_mod_polynomial(_POWERS[bit & 31], power)
        exponent >>= 1
        bit += 1
    return power


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

_GPU_CHUNK_SOURCE = r"""
__device__ unsigned int multiply_mod_polynomial(unsigned int a, unsigned int b) {
    unsigned int top = 1u << 31, product = 0;
    for (;;) {
        if (a & top) { product ^= b; if ((a & (top - 1u)) == 0) break; }
        top >>= 1;
        b = (b & 1u) ? (b >> 1) ^ CRC32C_POLYNOMIAL : b >> 1;
    }
    return product;
}

__device__ unsigned int byte_shift(long long bytes, const unsigned int* __restrict__ powers) {
    unsigned int power = 1u << 31;
    int bit = 3;
    for (long long exponent = bytes; exponent; exponent >>= 1, ++bit) {
        if (exponent & 1) power = multiply_mod_polynomial(powers[bit & 31], power);
    }
    return power;
}

__device__ unsigned int read_u32(const unsigned char* at) {
    return (unsigned int)at[0] | ((unsigned int)at[1] << 8) | ((unsigned int)at[2] << 16) | ((unsigned int)at[3] << 24);
}

extern "C" __global__
void crc32c_segments(const unsigned char* __restrict__ data, const long long* __restrict__ chunk_offset,
                     const long long* __restrict__ chunk_size, const long long* __restrict__ segment_start,
                     long long chunks, long long segment_bytes, const unsigned int* __restrict__ table,
                     unsigned int* __restrict__ segment_crc) {
    // One thread per segment: the full crc32c (initial and final xor all ones) of its bytes.
    __shared__ unsigned int lookup[256];
    for (int at = threadIdx.x; at < 256; at += blockDim.x) lookup[at] = table[at];
    __syncthreads();
    long long segments = segment_start[chunks];
    for (long long segment = (long long)blockIdx.x * blockDim.x + threadIdx.x; segment < segments;
         segment += (long long)gridDim.x * blockDim.x) {
        long long low = 0, high = chunks - 1;
        while (low < high) {
            long long middle = (low + high + 1) / 2;
            if (segment_start[middle] <= segment) low = middle; else high = middle - 1;
        }
        long long first = chunk_offset[low] + (segment - segment_start[low]) * segment_bytes;
        long long stop = min(first + segment_bytes, chunk_offset[low] + chunk_size[low]);
        unsigned int crc = 0xFFFFFFFFu;
        for (long long at = first; at < stop; ++at) crc = lookup[(crc ^ data[at]) & 0xFFu] ^ (crc >> 8);
        segment_crc[segment] = ~crc;
    }
}

extern "C" __global__
void verify_chunks(const unsigned char* __restrict__ data, const long long* __restrict__ chunk_offset,
                   const long long* __restrict__ chunk_size, const long long* __restrict__ segment_start,
                   long long segment_bytes, const unsigned int* __restrict__ powers,
                   const unsigned int* __restrict__ segment_crc, int chunk_rows,
                   int* __restrict__ error, long long* __restrict__ first_bad) {
    // One block per chunk: its segments' crc32c combine as a tree in shared memory (log2 of the
    // segment count steps, each shifting the left crc by the right range's bytes), then the
    // result is compared with the stored crc32c and the row size table with the payload size.
    extern __shared__ long long shared[];
    long long* length = shared;
    unsigned int* crc = (unsigned int*)(shared + blockDim.x);
    __shared__ unsigned long long table_total;
    __shared__ int table_failure;
    long long chunk = blockIdx.x;
    long long offset = chunk_offset[chunk], size = chunk_size[chunk];
    long long first = segment_start[chunk], count = segment_start[chunk + 1] - first;
    int thread = threadIdx.x;
    if (thread == 0) { table_total = 0; table_failure = 0; }
    if (thread < count) {
        crc[thread] = segment_crc[first + thread];
        length[thread] = min(segment_bytes, size - thread * segment_bytes);
    }
    __syncthreads();
    for (long long step = 1; step < count; step <<= 1) {
        if (thread % (2 * step) == 0 && thread + step < count) {
            crc[thread] = multiply_mod_polynomial(byte_shift(length[thread + step], powers), crc[thread]) ^ crc[thread + step];
            length[thread] += length[thread + step];
        }
        __syncthreads();
    }
    long long table_bytes = 4LL * chunk_rows;
    if (size >= table_bytes) {
        for (int row = thread; row < chunk_rows; row += blockDim.x) {
            unsigned int frame = read_u32(data + offset + 4LL * row);
            if (frame < 1u) atomicOr(&table_failure, 2);
            atomicAdd(&table_total, (unsigned long long)frame);
        }
    }
    __syncthreads();
    if (thread != 0) return;
    int failure = crc[0] != read_u32(data + offset + size) ? 1 : 0;
    if (!failure && (size < table_bytes || table_failure || (long long)table_total + table_bytes != size)) failure = 2;
    if (failure) { atomicOr(error, failure); atomicMin((unsigned long long*)first_bad, (unsigned long long)chunk); }
}

extern "C" __global__
void locate_frames(const unsigned char* __restrict__ data, const long long* __restrict__ chunk_offset,
                   const long long* __restrict__ chunk_size, int chunk_rows, const long long* __restrict__ wanted, long long rows,
                   long long samples, int sample_bytes, int* __restrict__ error,
                   long long* __restrict__ depth, long long* __restrict__ dictionary_offset, long long* __restrict__ slot_offset,
                   long long* __restrict__ exception_offset, long long* __restrict__ exception_count) {
    // One thread per wanted row: its frame from the chunk's size table, checked against the frame
    // size. A row that does not parse is flagged and given a one-byte dictionary and no
    // exceptions inside its chunk, so no decode ever reads or writes outside its buffers.
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    long long global = wanted[row], chunk = global / chunk_rows;
    int within = (int)(global % chunk_rows);
    long long offset = chunk_offset[chunk], size = chunk_size[chunk], table_bytes = 4LL * chunk_rows;
    int failure = 0;
    long long start = offset, stop = offset, bits = 0, dictionary = offset, slots = offset, exceptions = offset, count = 0;
    if (size < table_bytes) failure = 4;
    else {
        long long prefix = 0;
        for (int earlier = 0; earlier < within; ++earlier) prefix += read_u32(data + offset + 4LL * earlier);
        start = offset + table_bytes + prefix;
        stop = start + read_u32(data + offset + 4LL * within);
        if (stop <= start || stop > offset + size) failure = 4;
        else {
            bits = data[start];
            if (bits > MAXIMUM_DEPTH) failure = 4;
            else {
                dictionary = start + 1;
                slots = dictionary + (1LL << bits);
                exceptions = slots + (samples * bits + 7) / 8;
                long long remainder = stop - exceptions;
                if (remainder < 0 || remainder % (sample_bytes + 1)) failure = 4;
                else count = remainder / (sample_bytes + 1);
            }
        }
    }
    if (failure) {
        atomicOr(error, failure);
        bits = 0; dictionary = offset; slots = offset; exceptions = offset; count = 0;
    }
    depth[row] = bits; dictionary_offset[row] = dictionary; slot_offset[row] = slots;
    exception_offset[row] = exceptions; exception_count[row] = count;
}

extern "C" __global__
void scatter_checked_exceptions(const unsigned char* __restrict__ frames, const long long* __restrict__ exception_offset,
                                const long long* __restrict__ exception_count, unsigned char* __restrict__ out,
                                long long first_row, long long out_stride, int sample_bytes, long long samples,
                                int* __restrict__ error) {
    // scatter_row_exceptions, refusing (and flagging) a sample outside the row.
    long long row = first_row + blockIdx.x;
    long long count = exception_count[row];
    const unsigned char* sample_bytes_at = frames + exception_offset[row];
    const unsigned char* codes = sample_bytes_at + count * sample_bytes;
    unsigned char* target = out + row * out_stride;
    for (long long at = threadIdx.x; at < count; at += blockDim.x) {
        unsigned long long sample = 0;
        for (int byte = 0; byte < sample_bytes; ++byte) sample |= (unsigned long long)sample_bytes_at[at * sample_bytes + byte] << (8 * byte);
        if (sample < (unsigned long long)samples) target[sample] = codes[at];
        else atomicOr(error, 8);
    }
}
"""


CHUNK_FAILURES = {1: "fails its crc32c check", 2: "has a row size table that does not match its stored size",
                  4: "holds a row frame that does not fit its depth", 8: "names an exception sample outside the row"}
"""The device checks' failure bits, as ``read_rows_to_device`` reports them."""


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
        chunk_module = cupy.RawModule(
            code=_GPU_CHUNK_SOURCE.replace("CRC32C_POLYNOMIAL", f"{CRC32C_POLYNOMIAL:#x}u").replace("MAXIMUM_DEPTH", str(MAXIMUM_DEPTH))
        )
        self._crc_segments = chunk_module.get_function("crc32c_segments")
        self._verify = chunk_module.get_function("verify_chunks")
        self._locate = chunk_module.get_function("locate_frames")
        self._checked_exceptions = chunk_module.get_function("scatter_checked_exceptions")
        self._crc_table = cupy.asarray(_crc32c_table())
        self._powers = cupy.asarray(np.asarray(_POWERS, dtype=np.uint32))

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

    def decode_chunks(
        self, device_buffer: Any, chunk_offsets: I64Array, payload_sizes: I64Array, chunk_rows: int,
        rows: I64Array, sample_count: int, out: Any,
    ) -> tuple[Any, Any]:
        """Check the chunks at ``chunk_offsets`` in ``device_buffer`` and decode their ``rows``
        (chunk-major: staged chunk position * ``chunk_rows`` + row in chunk) into ``out``, all on the device.

        Each chunk's crc32c (the 4 bytes after its payload) is computed from segments, one thread
        each, and combined by one block per chunk as a tree, its size table is checked against its payload size, and
        every row's frame is located and checked; a row that does not parse decodes as zeros of
        its dictionary's first byte and never reaches outside its buffers. Queued on the current
        stream. Returns device ``(error, first_bad_chunk)``: nonzero ``error`` bits are
        ``CHUNK_FAILURES``, and the caller reads them once the stream has run.
        """
        cp = self.cupy
        wanted = np.asarray(rows, dtype=np.int64)
        rows = int(wanted.shape[0])
        if out.shape != (rows, sample_count) or out.dtype != cp.uint8 or out.strides[1] != 1:
            raise ValueError(f"out must be uint8 [{rows}, {sample_count}] with contiguous rows.")
        offsets = np.asarray(chunk_offsets, dtype=np.int64)
        sizes = np.asarray(payload_sizes, dtype=np.int64)
        chunks = int(offsets.shape[0])
        error = cp.zeros(1, dtype=cp.int32)
        first_bad = cp.full(1, np.iinfo(np.int64).max, dtype=cp.int64)
        if chunks == 0:
            return error, first_bad
        # the smallest segments that still let one block combine a whole chunk's
        segment_bytes = max(1, -(-int(sizes.max()) // self._threads))
        segment_start = np.concatenate(([0], np.cumsum(np.maximum(1, -(-sizes // segment_bytes))))).astype(np.int64)
        device_offsets, device_sizes, device_segment_start = (cp.asarray(values) for values in (offsets, sizes, segment_start))
        segment_crc = cp.empty(int(segment_start[-1]), dtype=cp.uint32)
        segments = int(segment_start[-1])
        self._crc_segments(
            (min(-(-segments // self._threads), self._resident_blocks),), (self._threads,),
            (device_buffer, device_offsets, device_sizes, device_segment_start, np.int64(chunks), np.int64(segment_bytes),
             self._crc_table, segment_crc),
        )
        self._verify(
            (chunks,), (self._threads,),
            (device_buffer, device_offsets, device_sizes, device_segment_start, np.int64(segment_bytes), self._powers,
             segment_crc, np.int32(chunk_rows), error, first_bad),
            shared_mem=self._threads * (np.dtype(np.int64).itemsize + np.dtype(np.uint32).itemsize),
        )
        if rows == 0:
            return error, first_bad
        sample_bytes = np.int32(exception_sample_dtype(sample_count).itemsize)
        depth, dictionary_offset, slot_offset, exception_offset, exception_count = (cp.empty(rows, dtype=cp.int64) for _ in range(5))
        self._locate(
            (-(-rows // self._threads),), (self._threads,),
            (device_buffer, device_offsets, device_sizes, np.int32(chunk_rows), cp.asarray(wanted), np.int64(rows),
             np.int64(sample_count), sample_bytes, error, depth, dictionary_offset, slot_offset, exception_offset, exception_count),
        )
        stride = np.int64(out.strides[0])
        blocks_per_row = max(1, min(-(-sample_count // self._threads), self._resident_blocks // min(rows, self._resident_blocks)))
        for first in range(0, rows, self._grid_rows):
            self._slots(
                (blocks_per_row, min(self._grid_rows, rows - first)), (self._threads,),
                (device_buffer, dictionary_offset, slot_offset, depth, out, np.int64(first), np.int64(sample_count), stride),
            )
        for first in range(0, rows, self._grid_blocks):
            self._checked_exceptions(
                (min(self._grid_blocks, rows - first),), (self._threads,),
                (device_buffer, exception_offset, exception_count, out, np.int64(first), stride, sample_bytes,
                 np.int64(sample_count), error),
            )
        return error, first_bad


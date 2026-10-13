"""CPU rowdict decoding at 50k/100k samples, checked against original codes.

Compares the original array decoder with direct compiled decoding. Timings
exclude encoding/JIT warmup. Traced memory covers Python/NumPy allocations,
not whole-process RSS. No real participant data are used.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc

import numpy as np

from sv_pgs import rowdict_codec as codec


def array_decode(buffer, frames, samples, out):
    sample_dtype = codec.exception_sample_dtype(samples)
    for bits in np.unique(frames.depth).tolist():
        members = np.flatnonzero(frames.depth == bits)
        size = 1 << bits
        dictionaries = buffer[frames.dictionary_offset[members, None] + np.arange(size)[None, :]]
        if not bits:
            out[members] = dictionaries
            continue
        width = int(codec.slot_bytes(samples, bits))
        packed = np.zeros((members.size, width + 1), dtype=np.uint16)
        packed[:, :width] = buffer[frames.slot_offset[members, None] + np.arange(width)[None, :]]
        byte, shift = np.divmod(np.arange(samples, dtype=np.int64) * bits, codec.BITS_PER_CODE)
        window = packed[:, byte] | (packed[:, byte + 1] << np.uint16(codec.BITS_PER_CODE))
        slot = (window >> shift.astype(np.uint16)) & np.uint16(size - 1)
        out[members] = np.take_along_axis(dictionaries, slot.astype(np.intp), axis=1)
    counts = frames.exception_count
    rows = np.repeat(np.arange(counts.size), counts)
    within = np.arange(rows.size) - (np.cumsum(counts) - counts)[rows]
    columns = codec._gather_uint(buffer, frames.exception_offset[rows] + within * sample_dtype.itemsize, sample_dtype)
    if np.any(columns >= samples):
        raise ValueError("invalid exception")
    out[rows, columns] = buffer[frames.exception_offset[rows] + counts[rows] * sample_dtype.itemsize + within]


def array_histograms(codes):
    offsets = codes.astype(np.int64) + codec.CODE_VALUES * np.arange(codes.shape[0])[:, None]
    return np.bincount(offsets.ravel(), minlength=codec.CODE_VALUES * codes.shape[0]).reshape(codes.shape[0], codec.CODE_VALUES)


def benchmark_histograms(codes, repeats):
    methods = [("array", array_histograms), ("compiled", codec._histograms)]
    expected = np.stack([np.bincount(row, minlength=codec.CODE_VALUES) for row in codes])
    peaks, times = {}, {name: [] for name, _ in methods}
    for name, method in methods:
        np.testing.assert_array_equal(method(codes), expected)
        tracemalloc.start()
        result = method(codes)
        _, peaks[name] = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del result
    for repeat in range(repeats):
        for name, method in methods[::1 if repeat % 2 == 0 else -1]:
            start = time.perf_counter()
            result = method(codes)
            times[name].append(time.perf_counter() - start)
            np.testing.assert_array_equal(result, expected)
            del result
    medians = {name: float(np.median(values)) for name, values in times.items()}
    print(json.dumps({"stage": "histogram", "samples": codes.shape[1], "rows": codes.shape[0],
                      "repeats": repeats, "bitwise_equal": True, "median_seconds": medians,
                      "speedup": medians["array"] / medians["compiled"],
                      "peak_traced_bytes_including_result": peaks}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[50_000, 100_000])
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.samples, args.rows, args.repeats) < 1:
        parser.error("sizes must be positive")
    rng = np.random.default_rng(182)
    for samples in args.samples:
        codes = np.empty((args.rows, samples), dtype=np.uint8)
        for row in range(args.rows):
            levels = min(1 << (row % (codec.MAXIMUM_DEPTH + 1)), 255)
            codes[row] = rng.integers(0, levels, size=samples, dtype=np.uint8)
            codes[row, [0, samples - 1]] = 254
        benchmark_histograms(codes, args.repeats)
        payload = codec.encode_chunk(codes)
        buffer = np.frombuffer(payload, dtype=np.uint8)
        frames = codec.row_frames(buffer, np.array([0]), np.array([len(payload)]), args.rows, samples, np.arange(args.rows))
        out = np.empty_like(codes)
        methods = [("array", array_decode), ("compiled", codec.decode_rows)]
        peaks = {}
        for name, method in methods:
            method(buffer, frames, samples, out)
            np.testing.assert_array_equal(out, codes)
            tracemalloc.start()
            method(buffer, frames, samples, out)
            _, peaks[name] = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        times = {name: [] for name, _ in methods}
        for repeat in range(args.repeats):
            for name, method in methods[::1 if repeat % 2 == 0 else -1]:
                start = time.perf_counter()
                method(buffer, frames, samples, out)
                times[name].append(time.perf_counter() - start)
                np.testing.assert_array_equal(out, codes)
        medians = {name: float(np.median(values)) for name, values in times.items()}
        print(json.dumps({"stage": "decode", "samples": samples, "rows": args.rows, "repeats": args.repeats,
                          "bitwise_equal": True, "median_seconds": medians,
                          "speedup": medians["array"] / medians["compiled"],
                          "peak_traced_scratch_bytes": peaks}), flush=True)


if __name__ == "__main__":
    main()

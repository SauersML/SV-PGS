"""The Gram-space sweep's time per band entry with the band held on the host (uploaded every sweep) and on the device.

A synthetic band of ``--blocks`` blocks of ``--width`` columns (chr22 bench-sim's Stage 0 blocks are 20k to 26k wide)
with float32 entries and diagonal n, a single-class prior, and three sweeps of each placement on one GPU; prints each
placement's seconds per sweep and per 10^9 band entries, which scale to any band by its entry count. Timing only: the
values are random [own-sim]."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sv_pgs.gram_space import GramBand


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sweeps", type=int, default=3)
    arguments = parser.parse_args()
    import cupy

    generator = np.random.default_rng(1)
    width, count, samples = arguments.width, arguments.blocks, arguments.samples

    def block(rows: int, columns: int, diagonal: bool) -> np.ndarray:
        values = (generator.standard_normal((rows, columns), dtype=np.float32) * np.float32(np.sqrt(samples) * 0.1))
        if diagonal:
            values = (values + values.T) * np.float32(0.5)
            values[np.arange(rows), np.arange(rows)] = np.float32(samples)
        return values

    within = [block(width, width, True) for _ in range(count)]
    cross = [block(width, width, False) for _ in range(count - 1)]
    entries = float(count * width * width + (count - 1) * width * width)
    members = count * width
    scores = generator.standard_normal(members) * np.sqrt(samples)
    grid = np.linspace(-12.0, 0.0, 60)
    log_density = np.log(np.full((1, grid.shape[0]), 1.0 / grid.shape[0]))
    member_blocks = tuple(np.arange(b * width, (b + 1) * width) for b in range(count))
    report = {"blocks": count, "width": width, "entries": entries}
    for placement in ("host", "device"):
        arrays_within = within if placement == "host" else [cupy.asarray(values) for values in within]
        arrays_cross = cross if placement == "host" else [cupy.asarray(values) for values in cross]
        band = GramBand.from_arrays(
            within=arrays_within, cross=arrays_cross, scores=scores, target_square=float(samples), sample_count=samples,
            residual_dimension=float(samples), working_bytes=1 << 30,
        )
        band.squares = np.full(members, float(samples))
        band._raw_squares = band.squares.copy()
        state = [np.zeros(members) for _ in range(5)]
        times = []
        for _sweep in range(arguments.sweeps):
            cupy.cuda.Stream.null.synchronize()
            started = time.perf_counter()
            band.sweep(
                cupy, member_blocks=member_blocks, group=np.arange(members), sign=np.ones(members), class_index=np.zeros(members, dtype=np.int64),
                log_density=log_density, scales=np.full(members, -3.0), grid=grid, noise=0.9, mean=state[0], variance=state[1],
                shift=state[2], third=state[3], fourth=state[4],
            )
            cupy.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - started)
        best = min(times[1:]) if len(times) > 1 else times[0]
        report[placement] = {"seconds": times, "per_billion_entries": best / (entries / 1e9)}
        del arrays_within, arrays_cross, band
        cupy.get_default_memory_pool().free_all_blocks()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

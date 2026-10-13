"""Measure row-kernel scheduling with identical compiled arithmetic.

Run ``python -m benchmarks.host_kernel_throughput``. Reports warmed timings for
the original worker count and the small-job scheduler; outputs must match bitwise.
"""
from __future__ import annotations

import argparse
import json
import time

import numba
import numpy as np

from sv_pgs.host_kernels import row_kernel
from sv_pgs.scale_mixture_ep import _LOST_BELOW, _moving_line_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rng = np.random.default_rng(71)
    results = []
    for rows, count in ((16, 1), (60, 15), (500, 15), (5000, 15)):
        grid = np.linspace(-9., 4., 33)
        density = rng.normal(size=(count, grid.size))
        density -= np.logaddexp.reduce(density, axis=1)[:, None]
        scale, slope, shift = rng.normal(size=(3, rows))
        steps, precision = np.linspace(-1., 1., count), np.exp(rng.normal(size=rows))
        out = np.empty((rows, count))
        arguments = (density, scale, slope, steps, grid, precision, shift, _LOST_BELOW, out)
        _moving_line_rows(*arguments)
        reference = out.copy()
        visits = rows * count * grid.size
        row_kernel(_moving_line_rows, visits, *arguments)
        np.testing.assert_array_equal(out, reference)
        record = {"rows": rows, "steps": count, "nodes": grid.size}
        for name, evaluate in (
            ("original", lambda: _moving_line_rows(*arguments)),
            ("scheduled", lambda: row_kernel(_moving_line_rows, visits, *arguments)),
        ):
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                evaluate()
                times.append(time.perf_counter() - start)
            record[name] = {"minimum_seconds": min(times), "median_seconds": float(np.median(times))}
        results.append(record)
    print(json.dumps({"workers": numba.get_num_threads(), "repeats": args.repeats,
                      "bitwise_equal": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()

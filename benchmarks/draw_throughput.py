"""Time posterior-draw generation against the original vectorized Philox path.

Run ``python -m benchmarks.draw_throughput``. Compilation is warmed before timing.
Every comparison requires bitwise identical outputs. This measures draw generation,
not complete fitting, scoring, or GPU throughput.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from unittest.mock import patch

import numpy as np

from sv_pgs import draw_laws


def reference_uniforms(ids, draws, key):
    words = draw_laws.philox4x32(
        (ids[:, None] & np.uint64(0xFFFFFFFF), ids[:, None] >> np.uint64(32),
         draws[None, :], np.zeros((1, draws.size), dtype=np.uint64)), key,
    )
    return draw_laws._uniform(words[0], words[1], False), draw_laws._uniform(words[2], words[3], True)


def measure(function, repeats):
    function()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - started)
        del result
    tracemalloc.start()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"minimum_seconds": min(times), "median_seconds": float(np.median(times)), "peak_traced_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.draws, args.repeats) < 1:
        parser.error("rows, draws and repeats must be positive")
    rng = np.random.default_rng(218)
    grid = np.linspace(-7., 1., 33)
    density = rng.normal(size=(4, grid.size))
    density -= np.logaddexp.reduce(density, axis=1)[:, None]
    law = draw_laws.ProductMixtureDraws(
        class_index=rng.integers(4, size=args.rows), log_scale=rng.normal(-1., .3, args.rows),
        log_density=density, log_variance_grid=grid,
        omega=rng.uniform(20., 200., size=(2, args.rows)), shift=rng.normal(0., 3., size=(2, args.rows)),
        weights=np.array([.6, .4]), key=(17, 29), draw_count=args.draws,
    )
    ids, draws = np.arange(args.rows, dtype=np.uint64), np.arange(args.draws, dtype=np.uint64)
    fused = draw_laws._member_uniforms
    for actual, expected in zip(fused(ids, draws, law.key), reference_uniforms(ids, draws, law.key)):
        np.testing.assert_array_equal(actual, expected)
    expected = law.tile(0, args.rows)
    with patch.object(draw_laws, "_member_uniforms", reference_uniforms):
        np.testing.assert_array_equal(law.tile(0, args.rows), expected)
    del expected
    results = {}
    for name, uniforms in (("reference", reference_uniforms), ("fused", fused)):
        with patch.object(draw_laws, "_member_uniforms", uniforms):
            results[name] = {
                "uniforms": measure(lambda: uniforms(ids, draws, law.key), args.repeats),
                "complete_draw_tile": measure(lambda: law.tile(0, args.rows), args.repeats),
            }
    print(json.dumps({"rows": args.rows, "draws": args.draws, "repeats": args.repeats,
                      "python": platform.python_version(), "numpy": np.__version__,
                      "machine": platform.machine(), "bitwise_equal": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()

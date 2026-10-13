"""Time a full-precision likelihood operator at 50k/100k synthetic samples.

Compares the old per-block operand preparation with the shared preparation.
Both execute the same genotype products and covariate projection. This is an
operator benchmark, not a complete fit. Run with --device cuda on a GPU host.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sv_pgs import dual_solve
from sv_pgs.code_products import CodeBlockTile
from sv_pgs.compute_budget import _try_import_cupy


class _Tiles:
    def __init__(self, samples, variants, block, xp, rng):
        self.array_module = xp
        self.sample_count = samples
        self.block_variant_indices = []
        self.tiles = []
        for first in range(0, variants, block):
            last = min(first + block, variants)
            codes = rng.integers(-127, 128, (last - first, samples), dtype=np.int8)
            means, scales = codes.mean(axis=1), codes.std(axis=1)
            self.block_variant_indices.append(np.arange(first, last))
            self.tiles.append(CodeBlockTile(xp.asarray(codes), xp.asarray(means), xp.asarray(scales), xp, 512 << 20))

    def iter_tiles(self):
        yield from enumerate(self.tiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[50_000, 100_000])
    parser.add_argument("--variants", type=int, default=1024)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(*args.samples, args.variants, args.block, args.columns, args.repeats) < 1:
        parser.error("sizes must be positive")
    xp = np if args.device == "cpu" else _try_import_cupy()
    if xp is None:
        parser.error("CUDA requested but no usable device is available")

    def sync():
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()

    for samples in args.samples:
        rng = np.random.default_rng(182)
        source = dual_solve.StreamedDualSource(_Tiles(samples, args.variants, args.block, xp, rng))
        variances = xp.asarray(rng.uniform(.5, 1.5, (args.variants, 1)) / args.variants)
        models = dual_solve.DualModels(xp.ones((samples, 1)), variances, xp.ones((samples, 1)), xp)
        values = xp.asarray(rng.normal(size=(samples, args.columns)))
        column_models = xp.zeros(args.columns, dtype=xp.int64)

        def old_work(start, stop, tile, shared, rows, image):
            product = tile.rmatmat(shared["left"])
            image += tile.matmat(rows["variances"][:, shared["column_models"]] * product)

        def original():
            left = models.sample_to_design(values, column_models)
            image = source.map_reduce(old_work, {"left": left, "column_models": column_models},
                                      {"variances": variances}, values.shape)
            return values + models.design_to_sample(image, column_models)

        def shared():
            return dual_solve.apply_operator(source, models, values, column_models, 0., dual_solve.PassCount(), "benchmark")

        expected, actual = original(), shared()
        # A diagnostic comparison tolerance, not a solver error certificate;
        # the regression suite separately checks residuals and posterior solves.
        difference = float(xp.max(xp.abs(expected - actual)))
        tolerance = (samples + args.variants) * np.finfo(float).eps * float(xp.max(xp.abs(expected)))
        if difference > tolerance:
            raise AssertionError("shared operator differs beyond the comparison tolerance")
        del expected, actual
        times = {"per_block": [], "shared": []}
        for repeat in range(args.repeats):
            methods = [("per_block", original), ("shared", shared)]
            for name, function in methods[::1 if repeat % 2 == 0 else -1]:
                sync()
                started = time.perf_counter()
                result = function()
                sync()
                times[name].append(time.perf_counter() - started)
                del result
        medians = {name: float(np.median(seconds)) for name, seconds in times.items()}
        print(json.dumps({"samples": samples, "variants": args.variants, "blocks": len(source.block_bounds),
                          "columns": args.columns, "device": args.device, "median_seconds": medians,
                          "speedup": medians["per_block"] / medians["shared"],
                          "max_difference": difference, "comparison_tolerance": tolerance}), flush=True)


if __name__ == "__main__":
    main()

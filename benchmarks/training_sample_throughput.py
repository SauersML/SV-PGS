"""Measure training-only genotype products at a realistic sample count.

``python -m benchmarks.training_sample_throughput --samples 100000`` compares
the original full-sample operator with zero-weight nontraining rows against
the compact operator. Use ``--device cuda`` on a CUDA machine. Both use the
existing float64-accuracy products; this is a kernel benchmark, not a study
runtime or cost estimate. Synthetic hard calls are generated locally.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sv_pgs.code_products import CodeBlockTile
from sv_pgs.compute_budget import _try_import_cupy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--variants", type=int, default=1024)
    parser.add_argument("--training-fraction", type=float, default=.72)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.samples, args.variants, args.columns, args.repeats, args.workspace_mib) < 1 or not 0 < args.training_fraction <= 1:
        parser.error("sizes must be positive and training fraction must be in (0, 1]")
    selected_count = int(args.samples * args.training_fraction)
    if selected_count < 2:
        parser.error("at least two training samples are required")
    xp = np if args.device == "cpu" else _try_import_cupy()
    if xp is None:
        parser.error("CUDA was requested but no usable CUDA device is available")
    rng = np.random.default_rng(718)
    selected = np.sort(rng.choice(args.samples, selected_count, replace=False))
    codes = rng.integers(-1, 2, size=(args.variants, args.samples), dtype=np.int8) * np.int8(127)
    training_codes = np.ascontiguousarray(codes[:, selected])
    means = training_codes.mean(axis=1)
    scales = training_codes.std(axis=1)
    if np.any(scales == 0):
        parser.error("synthetic training data contain a monomorphic row; use more samples")
    left = rng.normal(size=(selected_count, args.columns))
    full_left = np.zeros((args.samples, args.columns))
    full_left[selected] = left
    right = rng.normal(size=(args.variants, args.columns))
    width = args.workspace_mib * 1024**2
    tiles = [CodeBlockTile(xp.asarray(values), xp.asarray(means), xp.asarray(scales), xp, width)
             for values in (codes, training_codes)]
    operands = [xp.asarray(values) for values in (full_left, left)]
    device_right = xp.asarray(right)

    def sync():
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()

    def host(values):
        return values if xp is np else xp.asnumpy(values)

    # A conservative float64 summation bound: gamma_n times the sum of
    # absolute standardized terms, allowing both products' centering/scaling.
    unit = np.finfo(np.float64).eps / 2
    depth = args.samples + args.variants + 4
    gamma = depth * unit / (1 - depth * unit)
    max_standardized = (127 + np.abs(means)) / scales
    full_back = host(tiles[0].rmatmat(operands[0]))
    compact_back = host(tiles[1].rmatmat(operands[1]))
    back_bound = 2 * gamma * max_standardized[:, None] * np.abs(left).sum(axis=0)
    full_forward = host(tiles[0].matmat(device_right))[selected]
    compact_forward = host(tiles[1].matmat(device_right))
    forward_bound = 2 * gamma * (max_standardized @ np.abs(right))[None, :]
    if not (np.all(np.abs(full_back - compact_back) <= back_bound)
            and np.all(np.abs(full_forward - compact_forward) <= forward_bound)):
        raise AssertionError("compact products exceeded float64 rounding bounds")
    times = {"full": [], "compact": []}
    # Alternate order to reduce warm-cache and transient-load bias.
    for repeat in range(args.repeats):
        for position in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            sync()
            started = time.perf_counter()
            backward = tiles[position].rmatmat(operands[position])
            forward = tiles[position].matmat(device_right)
            sync()
            times[("full", "compact")[position]].append(time.perf_counter() - started)
            del backward, forward
    print(json.dumps({"device": args.device, "samples": args.samples, "training_samples": selected_count,
                      "variants": args.variants, "columns": args.columns, "repeats": args.repeats,
                      "workspace_bytes": width, "accuracy_within_rounding_bound": True,
                      "maximum_backward_difference": float(np.max(np.abs(full_back - compact_back))),
                      "maximum_forward_difference": float(np.max(np.abs(full_forward - compact_forward))),
                      "signed_code_bytes": dict(zip(("full", "compact"), [int(tile.aligned_codes.nbytes) for tile in tiles])),
                      "median_seconds": {name: float(np.median(values)) for name, values in times.items()}}, indent=2))


if __name__ == "__main__":
    main()

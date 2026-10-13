"""Compare fused operand preparation with the original array recurrence.

Synthetic operands at 50k/100k samples; reports kernel-preparation time, not
whole-fit speed. CUDA runs fail explicitly if a GPU is unavailable. Every
reported case must first agree bit for bit with the independent reference.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sv_pgs.code_products import DIGIT_BITS, OPERAND_DIGITS, operand_digits, _column_max
from sv_pgs.compute_budget import _try_import_cupy


def reference(values, xp, count):
    rows, columns = values.shape
    maximum = _column_max(xp.abs(values), xp)
    scale = xp.exp2((DIGIT_BITS * count - 2) - xp.ceil(xp.log2(xp.where(maximum > 0, maximum, 1.))))
    integers = xp.rint(values * scale).astype(xp.int64)
    digits = xp.empty((rows, count * columns), dtype=xp.int8, order="F")
    half = 1 << (DIGIT_BITS - 1)
    for digit in range(count):
        low = ((integers + half) & ((1 << DIGIT_BITS) - 1)) - half
        digits[:, digit * columns:(digit + 1) * columns] = low
        integers = (integers - low) >> DIGIT_BITS
    return digits, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[50_000, 100_000])
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--digits", type=int, default=OPERAND_DIGITS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(*args.samples, args.columns, args.repeats) < 1 or not 1 <= args.digits <= OPERAND_DIGITS:
        parser.error("sizes must be positive and digits between 1 and OPERAND_DIGITS")
    xp = np if args.device == "cpu" else _try_import_cupy()
    if xp is None:
        parser.error("CUDA was requested but no usable CUDA device is available")

    def sync():
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()

    rng = np.random.default_rng(44)
    for samples in args.samples:
        values = xp.asarray(rng.normal(size=(samples, args.columns)))
        expected = reference(values, xp, args.digits)
        actual = operand_digits(values, xp, args.digits)
        assert all(bool(xp.array_equal(left, right)) for left, right in zip(expected, actual))
        del expected, actual
        timings = {"reference": [], "fused": []}
        for repeat in range(args.repeats):
            operations = [("reference", reference), ("fused", operand_digits)]
            for name, operation in operations[::1 if repeat % 2 == 0 else -1]:
                sync()
                started = time.perf_counter()
                result = operation(values, xp, args.digits)
                sync()
                timings[name].append(time.perf_counter() - started)
                del result
        medians = {name: float(np.median(times)) for name, times in timings.items()}
        print(json.dumps({"samples": samples, "columns": args.columns, "digits": args.digits,
                          "device": args.device, "repeats": args.repeats, "bitwise_equal": True,
                          "median_seconds": medians, "speedup": medians["reference"] / medians["fused"],
                          "dense_integer_scratch_bytes_fused": 0}), flush=True)


if __name__ == "__main__":
    main()

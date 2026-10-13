"""Compare tied-member indexing with the former full-sort implementation.

Run ``python -m benchmarks.tie_indexing --members 2000000``. Timings describe
this operation only; they are not end-to-end model or GPU speedup estimates.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc

import numpy as np

from sv_pgs.tie_members import TieGroups, tied_groups


def full_sort(ties: TieGroups) -> list[np.ndarray]:
    """The previous implementation, retained here as a performance reference."""
    sizes = np.bincount(ties.group, minlength=ties.group_count)
    order = np.argsort(ties.group, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    return [order[bounds[group]:bounds[group + 1]] for group in np.flatnonzero(sizes > 1)]


def measure(function, ties: TieGroups, repeats: int) -> dict:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(ties)
        elapsed.append(time.perf_counter() - start)
        del result
    # Measure allocations separately; tracing itself significantly slows Python.
    tracemalloc.start()
    try:
        result = function(ties)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"minimum_seconds": min(elapsed), "median_seconds": float(np.median(elapsed)),
            "peak_traced_bytes": peak, "tied_groups": len(result)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, default=2_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.members < 2 or arguments.repeats < 1:
        parser.error("members must be at least two and repeats must be positive")
    results = []
    for stride in (0, 100, 4):
        group = np.arange(arguments.members)
        if stride:
            group[1::stride] -= 1
        ties = TieGroups(group, np.ones(arguments.members), arguments.members)
        expected, actual = full_sort(ties), tied_groups(ties)
        if len(expected) != len(actual) or any(not np.array_equal(a, b) for a, b in zip(expected, actual)):
            raise AssertionError("optimized group membership differs from the reference")
        del expected, actual
        results.append({"tie_stride": stride, "reference": measure(full_sort, ties, arguments.repeats),
                        "optimized": measure(tied_groups, ties, arguments.repeats)})
    print(json.dumps({"operation": "tied_member_indexing", "members": arguments.members,
                      "repeats": arguments.repeats, "python": platform.python_version(),
                      "numpy": np.__version__, "machine": platform.machine(), "results": results}, indent=2))


if __name__ == "__main__":
    main()

"""Scheduling of independent-row Numba kernels; no change to their arithmetic."""
from __future__ import annotations

import threading
import time

import numba


class _Schedules(threading.local):
    def __init__(self):
        self.choices = {}


_schedules = _Schedules()


def _one_worker(kernel, workers, arguments):
    numba.set_num_threads(1)
    try:
        return kernel(*arguments)
    finally:
        numba.set_num_threads(workers)


def row_kernel(kernel, node_visits: int, *arguments):
    """Measure one worker against the caller's pool for each new workload shape.

    Numba's thread mask is local to the calling thread. This changes neither
    the compiled kernel nor any within-row summation or downstream reduction.
    Kernels must completely overwrite their outputs on every call. After an
    untimed warmup, time both schedules (including mask changes) and remember
    the faster one for that kernel, worker count and array shapes. No arrays
    are retained, and the caller's worker count is never increased. Timing
    selects only a schedule for identical arithmetic, never an estimator.
    """
    workers = numba.get_num_threads()
    if workers == 1:
        return kernel(*arguments)
    shapes = tuple((argument.shape, argument.dtype.str) for argument in arguments if hasattr(argument, "shape"))
    key = (kernel, workers, node_visits, shapes)
    single = _schedules.choices.get(key)
    if single is None:
        kernel(*arguments)  # compilation and pool initialization are not timed
        started = time.perf_counter()
        _one_worker(kernel, workers, arguments)
        single_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = kernel(*arguments)
        parallel_seconds = time.perf_counter() - started
        _schedules.choices[key] = single_seconds < parallel_seconds
        return result
    return _one_worker(kernel, workers, arguments) if single else kernel(*arguments)

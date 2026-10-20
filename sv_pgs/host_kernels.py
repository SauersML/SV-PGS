"""Scheduling of independent-row Numba kernels; no change to their arithmetic."""
from __future__ import annotations

import threading
import time

import numba


class _Schedules(threading.local):
    def __init__(self):
        self.choices = {}


_schedules = _Schedules()

# Numba's workqueue threading layer, the one it falls back to on a host with neither TBB nor GNU OpenMP (the Batch VM
# image has no libgomp), is not thread-safe: two threads launching parallel kernels at once abort the process ("Nested
# parallel kernel launch detected", exit 134) or crash it (exit 139), as the stage-0 host pool's code conversions did
# in the All of Us fit. Every parallel launch holds this lock; each launch uses every worker anyway.
_LAUNCH = threading.Lock()


def launch(kernel, *arguments):
    """``kernel(*arguments)`` with no other parallel kernel running in the process."""
    with _LAUNCH:
        return kernel(*arguments)


def _one_worker(kernel, workers, arguments):
    numba.set_num_threads(1)
    try:
        return launch(kernel, *arguments)
    finally:
        numba.set_num_threads(workers)


def row_kernel(kernel, node_visits: int, *arguments):
    """Measure one worker against the caller's pool for each new workload shape.

    Numba's thread mask is local to the calling thread. This changes neither
    the compiled kernel nor any within-row summation or downstream reduction.
    Kernels must completely overwrite their outputs on every call. After an
    untimed warmup, time both schedules (including mask changes) and remember
    the faster one for that kernel, worker count and the octave of the work
    and of every array extent. No arrays are retained, and the caller's worker
    count is never increased. Timing selects only a schedule for identical
    arithmetic, never an estimator.

    Exact shapes and visit counts as the key retimed nearly every call of a
    kernel whose step count changes per call (a line search's), and a retime
    runs the kernel three times, once on one worker: a 16-core fit spent hours
    on one core there (bench-sim chr22, 5,000 training samples).
    """
    workers = numba.get_num_threads()
    if workers == 1:
        return launch(kernel, *arguments)
    shapes = tuple((tuple(int(extent).bit_length() for extent in argument.shape), argument.dtype.str) for argument in arguments if hasattr(argument, "shape"))
    key = (kernel, workers, int(node_visits).bit_length(), shapes)
    single = _schedules.choices.get(key)
    if single is None:
        launch(kernel, *arguments)  # compilation and pool initialization are not timed
        started = time.perf_counter()
        _one_worker(kernel, workers, arguments)
        single_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = launch(kernel, *arguments)
        parallel_seconds = time.perf_counter() - started
        _schedules.choices[key] = single_seconds < parallel_seconds
        return result
    return _one_worker(kernel, workers, arguments) if single else launch(kernel, *arguments)

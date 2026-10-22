"""The first CUDA error's operation trail, enabled by ``SVPGS_GPU_GUARD`` before importing the device code.

Five live attempts ended in cudaErrorLaunchFailure, first reported by an event or memory destructor, with no
operation named [real, whole-genome EB and hypothyroidism]. The last 128 operations, their threads and the stack
where the error surfaces are logged once. 0 (the default) leaves the functions themselves in place; 1 records the
trail, and sync or harden also checks the traced operations' devices. This diagnoses a fault, not its cause.
"""
from __future__ import annotations

from collections import deque
import functools
import os
import sys
import threading
import time
import traceback
from typing import Any, Callable

from sv_pgs.progress import log

TRAIL = 128
"""Operations in the live round's fault report (``SVPGS_GPU_GUARD``)."""
_MODE = os.environ.get("SVPGS_GPU_GUARD", "0").strip().lower()
_TRAIL: deque = deque(maxlen=TRAIL)
_LOCK = threading.Lock()
_REPORTED = False
_CUDA_ERRORS = frozenset({"CUDARuntimeError", "CUDADriverError", "CUBLASError", "CUSOLVERError", "CUSPARSEError", "CURANDError", "NVRTCError"})


def is_cuda_error(error: BaseException) -> bool:
    """Whether ``error`` is a CUDA runtime, library or compilation error, without importing CuPy."""
    return any(kind.__name__ in _CUDA_ERRORS for kind in type(error).__mro__)


def report(where: str, error: BaseException) -> None:
    """Log the first CUDA error with its operation trail and Python stack; preserve the original exception."""
    global _REPORTED
    if not is_cuda_error(error):
        return
    with _LOCK:
        if _REPORTED:
            return
        _REPORTED = True
        entries = list(_TRAIL)
    lines = [f"gpu guard: first CUDA error, {type(error).__name__}: {error}; surfaced at {where}, thread {threading.current_thread().name}"]
    if entries:
        last = entries[-1][0]
        lines.append(f"gpu guard: the last {len(entries)} traced GPU operations, oldest first:")
        lines.extend(f"gpu guard:   -{last - moment:.3f} s [{thread}] {label}" for moment, thread, label in entries)
    lines.append("gpu guard: the Python stack where it surfaced:")
    stack = traceback.format_stack()[:-1] + traceback.format_tb(error.__traceback__)
    lines.extend(f"gpu guard:   {line}" for frame in stack for line in frame.rstrip().splitlines())
    for line in lines:
        try:
            log(line)
        except Exception:
            pass


def traced(function: Callable) -> Callable:
    """Record ``function`` and report a CUDA error through it; return the function itself when the guard is off."""
    if _MODE == "0":
        return function

    name = getattr(function, "__name__", type(function).__name__)
    qualified = getattr(function, "__qualname__", name)

    @functools.wraps(function)
    def call(*arguments, **keywords):
        if name == "map_reduce":
            if len(arguments) > 1:
                arguments = (arguments[0], traced(arguments[1]), *arguments[2:])
            else:
                keywords = {**keywords, "work": traced(keywords["work"])}
        owner = arguments[0] if arguments else None
        cupy = getattr(owner, "_cupy", getattr(owner, "cupy", owner))
        device = getattr(owner, "_device_id", getattr(owner, "device_id", None))
        plan = getattr(owner, "plan", None)
        devices = tuple(plan.device_ids) if plan is not None else (() if device is None else (device,))
        if name == "moved":
            devices = (arguments[2] if len(arguments) > 2 else keywords["device_id"],)
        shapes = [getattr(getattr(value, "left", value), "shape", None) for value in arguments]
        label = f"{qualified} devices={devices} shapes={shapes}"
        if isinstance(owner, int) and len(arguments) > 1:
            label += f" block [{owner}:{arguments[1]})"
        with _LOCK:
            _TRAIL.append((time.monotonic(), threading.current_thread().name, label))
        try:
            if name == "settle" and _MODE in ("sync", "harden"):
                factory = getattr(getattr(cupy, "cuda", None), "Device", None)
                if factory is not None:
                    for device_id in devices:
                        factory(int(device_id)).synchronize()
            result = function(*arguments, **keywords)
            if _MODE in ("sync", "harden"):
                factory = getattr(getattr(cupy, "cuda", None), "Device", None)
                if factory is not None:
                    for device_id in devices:
                        factory(int(device_id)).synchronize()
            return result
        except BaseException as error:
            report(label, error)
            raise

    return call


def _unraisable(arguments: Any) -> None:
    """Report a CUDA error ignored in a destructor and preserve Python's previous handler."""
    error = getattr(arguments, "exc_value", None)
    if error is not None:
        report(f"{arguments.err_msg or 'Exception ignored in'} {type(arguments.object).__qualname__}", error)
    _PREVIOUS_UNRAISABLE(arguments)


_PREVIOUS_UNRAISABLE = sys.unraisablehook
if _MODE != "0":
    sys.unraisablehook = _unraisable

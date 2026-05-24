"""N-GPU block scheduler for LD-block dispatch.

Phase 3 of the LD-block / N-GPU rewrite. SV-PGS partitions variants into
LD blocks (~1700 small chunks); each block matvec is independent and
trivially parallel across N visible GPUs. This module provides a
round-robin dispatcher that works for N=1, 2, 4, 8, ..., transparently
falls back to CPU when cupy is unavailable, and never sets
``CUDA_VISIBLE_DEVICES`` (the process is allowed to see every device).

Concurrency model
-----------------
* One cupy stream per device per scheduler instance — created lazily on
  first ``device_context`` use so import is cheap and the scheduler does
  not allocate CUDA state until it is actually exercised.
* ``device_context(d)`` pushes the device, enters its stream, and pops
  both on exit.
* ``synchronize()`` waits on every per-device stream then issues
  ``cupy.cuda.runtime.deviceSynchronize()`` per device for paranoia
  (matches the wave-5713d85 CG cache sync pattern).
* A scheduler instance is *not* thread-safe: hold one per fit and let
  the surrounding pipeline drive it from a single Python thread.

CPU fallback
------------
* ``GPUScheduler.detect()`` returns a CPU-only scheduler
  (``device_ids=(-1,)``) when cupy is missing or reports zero devices.
* ``device_context(-1)`` is a no-op contextmanager, ``synchronize`` is a
  no-op, and ``aggregate_to_host`` simply moves any cupy arrays back via
  ``cupy.asnumpy`` or passes numpy through unchanged.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from sv_pgs._typing import NDArray

__all__ = ["DeviceAssignment", "GPUScheduler"]


@dataclass(frozen=True)
class DeviceAssignment:
    """Per-block dispatch record produced by :meth:`GPUScheduler.assign`."""

    block_id: int
    device_id: int  # cupy device id, or -1 for CPU fallback
    stream_id: int  # per-device stream slot (always 0 in current design)


def _probe_cupy() -> tuple[Any | None, int]:
    """Return (cupy_module_or_None, device_count). Never raises."""
    try:
        import cupy  # type: ignore[import-not-found]
    except Exception:
        return None, 0
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return cupy, 0
    return cupy, max(0, count)


class GPUScheduler:
    """Round-robin block dispatch across N visible cupy devices.

    Usage::

        sched = GPUScheduler.detect()
        for assignment in sched.assign(block_ids):
            with sched.device_context(assignment.device_id):
                # compute on this device, results land in per_device[d]
                ...
        sched.synchronize()
        host_sum = sched.aggregate_to_host(per_device)

    Notes
    -----
    * Round-robin: block ``b`` → device ``device_ids[b % N]``.
    * Streams are created on first ``device_context`` call per device.
    * Not thread-safe; create one scheduler per fit.
    """

    def __init__(self, device_ids: tuple[int, ...], cupy: Any | None) -> None:
        if not device_ids:
            raise ValueError("GPUScheduler requires at least one device id")
        self._device_ids: tuple[int, ...] = tuple(device_ids)
        self._cupy = cupy if device_ids != (-1,) else None
        # Lazy per-device streams: device_id -> cupy.cuda.Stream
        self._streams: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> "GPUScheduler":
        """Probe via cupy. If 0 devices or cupy unavailable, returns a
        CPU-only scheduler with ``device_ids=(-1,)``."""
        cupy, count = _probe_cupy()
        if cupy is None or count <= 0:
            return cls(device_ids=(-1,), cupy=None)
        return cls(device_ids=tuple(range(count)), cupy=cupy)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def device_count(self) -> int:
        return len(self._device_ids)

    @property
    def device_ids(self) -> tuple[int, ...]:
        return self._device_ids

    @property
    def is_cpu_fallback(self) -> bool:
        return self._device_ids == (-1,)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def assign(self, block_ids: Iterable[int]) -> Iterator[DeviceAssignment]:
        """Round-robin: block ``b`` is sent to ``device_ids[b % N]``.

        ``stream_id`` is always 0 in the current single-stream-per-device
        design but is reserved in the dataclass for future multi-stream
        experiments.
        """
        n = len(self._device_ids)
        for b in block_ids:
            yield DeviceAssignment(
                block_id=int(b),
                device_id=self._device_ids[int(b) % n],
                stream_id=0,
            )

    @contextmanager
    def device_context(self, device_id: int) -> Iterator[None]:
        """Enter ``cupy.cuda.Device(device_id)`` + per-device stream.

        No-op for the CPU-fallback sentinel ``device_id == -1``.
        """
        if device_id == -1 or self._cupy is None:
            yield
            return
        if device_id not in self._device_ids:
            raise ValueError(
                f"device_id={device_id} not in scheduler devices {self._device_ids}"
            )
        cupy = self._cupy
        device = cupy.cuda.Device(device_id)
        device.__enter__()
        try:
            stream = self._streams.get(device_id)
            if stream is None:
                stream = cupy.cuda.Stream(non_blocking=True)
                self._streams[device_id] = stream
            stream.__enter__()
            try:
                yield
            finally:
                stream.__exit__(None, None, None)
        finally:
            device.__exit__(None, None, None)

    def synchronize(self) -> None:
        """Wait for every per-device stream, then deviceSynchronize per device.

        No-op for the CPU-fallback scheduler.
        """
        if self._cupy is None:
            return
        cupy = self._cupy
        for device_id in self._device_ids:
            if device_id == -1:
                continue
            stream = self._streams.get(device_id)
            with cupy.cuda.Device(device_id):
                if stream is not None:
                    stream.synchronize()
                try:
                    cupy.cuda.runtime.deviceSynchronize()
                except Exception:
                    # Paranoia call; ignore drivers that disallow it post-stream-sync.
                    pass

    # ------------------------------------------------------------------
    # aggregation
    # ------------------------------------------------------------------

    def aggregate_to_host(self, per_device_arrays: dict[int, Any]) -> NDArray:
        """Move per-device arrays to host (numpy) and sum.

        Accepts a mapping of ``device_id -> array`` where arrays may be
        cupy or numpy. Returns the elementwise sum as a numpy array.
        Raises ``ValueError`` if the mapping is empty.
        """
        if not per_device_arrays:
            raise ValueError("aggregate_to_host requires at least one per-device array")
        cupy = self._cupy
        host_arrays: list[NDArray] = []
        for _, arr in per_device_arrays.items():
            if cupy is not None and isinstance(arr, getattr(cupy, "ndarray", ())):
                host_arrays.append(cupy.asnumpy(arr))
            else:
                host_arrays.append(np.asarray(arr))
        result = host_arrays[0].copy()
        for extra in host_arrays[1:]:
            result = result + extra
        return result

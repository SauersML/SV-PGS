"""GPUDirect Storage (cuFile) probe and direct device read helpers.

This module is an optional fast path for cold-loading bitpacked PLINK BED bytes
directly from NVMe into GPU HBM via NVIDIA's cuFile API, bypassing host RAM.

The Python binding used here is ``kvikio`` (https://github.com/rapidsai/kvikio).
It is an **optional dependency**: if ``kvikio`` is not installed, or if the
runtime is in cuFile "compat mode" (i.e. real GPUDirect Storage is unavailable
and kvikio would silently fall back to a POSIX read + host bounce buffer), then
:func:`gpudirect_available` returns ``False`` and callers are expected to
gracefully degrade to pinned-RAM staging via ``mmap_reader`` / ``preadv``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-only
    import cupy as cp  # noqa: F401


@functools.lru_cache(maxsize=1)
def gpudirect_available() -> bool:
    """Return True iff a real (non-compat) cuFile / GDS path is usable.

    Detection strategy:
      1. Attempt a lazy ``import kvikio`` (and ``kvikio.defaults``).
      2. Query ``kvikio.defaults.compat_mode_enabled``. kvikio exposes this as
         either a property or a zero-arg callable depending on version; we
         handle both. When compat mode is enabled, kvikio is doing a POSIX
         read into a host bounce buffer, which is not GPUDirect Storage, so
         we report False.
      3. Any ImportError / AttributeError / runtime error -> False.

    The result is cached for the lifetime of the process.
    """
    try:
        import kvikio  # noqa: F401
        from kvikio import defaults as _kv_defaults
    except Exception:
        return False

    try:
        compat = _kv_defaults.compat_mode_enabled
        # Newer kvikio exposes this as a callable; older as a plain attribute.
        if callable(compat):
            compat = compat()
        return not bool(compat)
    except Exception:
        return False


def cufile_read_to_device(
    path: "Path | str",
    device_buffer: "cp.ndarray",
    offset: int,
    count: int,
) -> None:
    """Read ``count`` bytes from ``path`` at byte ``offset`` directly into
    ``device_buffer`` (a CuPy device array) using cuFile / GPUDirect Storage.

    Parameters
    ----------
    path:
        Filesystem path to read from. Must reside on a GDS-capable filesystem
        for the true DMA path; otherwise the caller should have already
        detected via :func:`gpudirect_available` and chosen a different path.
    device_buffer:
        A CuPy ``ndarray`` whose underlying device memory is the destination.
        Must have at least ``count`` bytes of capacity starting at element 0.
    offset:
        Byte offset within the file to start reading from.
    count:
        Number of bytes to read.

    Raises
    ------
    RuntimeError
        If :func:`gpudirect_available` is False (kvikio missing or compat mode).
    """
    if not gpudirect_available():
        raise RuntimeError(
            "cuFile / GPUDirect Storage not available "
            "(kvikio missing or running in compat mode); "
            "callers should fall back to pinned-RAM staging."
        )

    import kvikio  # lazy

    path_str = str(path)
    # device_buffer is a CuPy ndarray; kvikio.CuFile.pread accepts any
    # CUDA-array-interface buffer. Slice to exactly `count` bytes so kvikio
    # reads no more than requested.
    view = device_buffer.view(dtype="uint8").ravel()[:count]

    with kvikio.CuFile(path_str, "r") as f:
        future = f.pread(view, size=count, file_offset=int(offset))
        # pread returns a future-like object; .get() blocks until the DMA
        # has completed and returns the number of bytes read.
        n_read = future.get() if hasattr(future, "get") else int(future)

    if int(n_read) != int(count):
        raise RuntimeError(
            f"cufile_read_to_device: short read from {path_str} "
            f"(requested {count} bytes at offset {offset}, got {n_read})"
        )

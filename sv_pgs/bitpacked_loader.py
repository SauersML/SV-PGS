"""PLINK BED → device-resident bitpacked CuPy buffer loader.

Implements ``load_bed_to_bitpacked_device`` per ``BITPACKED_SPEC.md``:

- Reads PLINK 1.9 BED bytes via ``sv_pgs.plink.open_bed`` (read-only utility).
- Optionally gathers a variant subset using the coalesced indexed reader.
- Optionally rebitpacks to a sample subset on CPU before device upload.
- Stages bytes through pinned host memory and DMAs to device asynchronously.
- Computes per-variant mean/std with the bitpacked screening kernel when
  they are not supplied by the caller.
- Wraps the resulting device buffer in a ``BitpackedDeviceMatrix``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sv_pgs.gcsfuse_staging import is_gcsfuse_path
from sv_pgs.path_policy import assert_safe_for_purpose
from sv_pgs.plink import (
    _BYTE_DECODE_LUT_A1,
    _BYTE_DECODE_LUT_A2,
    _ENCODE_LOOKUP_A1,
    _bytes_per_variant,
    open_bed,
)

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix

# Stream all-variant reads in chunks no larger than this many bytes to bound
# host-side peak RSS when the BED is very large.
_FULL_READ_CHUNK_BYTES: int = 256 * 1024 * 1024

# Minimum chunk size when streaming from gcsfuse — large requests are essential
# to saturate network bandwidth (~200 MB/s sustained). The default 256 MB cap
# already exceeds this floor, but the floor protects against the small
# ``bytes_per_variant`` chunked-variant integer-division path.
_GCSFUSE_MIN_CHUNK_BYTES: int = 16 * 1024 * 1024

# Below this total payload size, parallel range reads are not worth it. gcsfuse
# small-file reads complete in well under a second on a single stream and the
# thread-pool spin-up cost dominates. Tuned to 1 GB (≈ 5 s at 200 MB/s single-
# stream — anything shorter doesn't justify multiplexing).
_PARALLEL_GCSFUSE_THRESHOLD_BYTES: int = 1 * 1024 * 1024 * 1024


def _fadvise_sequential(fd: int) -> None:
    """Hint POSIX_FADV_SEQUENTIAL on ``fd``. Best-effort; Linux-only."""
    fadvise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_SEQUENTIAL", None)
    if fadvise is None or advice is None:
        return
    try:
        fadvise(fd, 0, 0, advice)
    except OSError:
        pass


def _fadvise_dontneed(fd: int) -> None:
    """Hint POSIX_FADV_DONTNEED on ``fd`` so the page cache can be reclaimed
    after a one-shot read (we already hold the data in HBM)."""
    fadvise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or advice is None:
        return
    try:
        fadvise(fd, 0, 0, advice)
    except OSError:
        pass


def _require_cupy() -> Any:
    """Import cupy lazily and raise a clear error if it is unavailable."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without cupy
        raise RuntimeError(
            "load_bed_to_bitpacked_device requires CuPy to upload bitpacked "
            "PLINK BED bytes to the GPU, but `import cupy` failed: "
            f"{exc}. Install a CuPy build matching the local CUDA toolkit."
        ) from exc
    return cp


def _resolve_stream(cp: Any, stream: Any) -> Any:
    """Return a usable CuPy stream object (current stream if ``stream`` is None)."""
    if stream is None:
        return cp.cuda.Stream.null
    return stream


def _allocate_pinned(cp: Any, nbytes: int) -> tuple[Any, np.ndarray]:
    """Allocate a pinned-host uint8 staging buffer and return (mem, numpy_view)."""
    if nbytes <= 0:
        return None, np.empty((0,), dtype=np.uint8)
    pinned_mem = cp.cuda.alloc_pinned_memory(nbytes)
    # Wrap the pinned allocation in a numpy view so the caller can fill it
    # with positional reads / rebitpacked bytes without an extra copy.
    host_view = np.frombuffer(pinned_mem, dtype=np.uint8, count=nbytes)
    return pinned_mem, host_view


def _h2d_async(
    cp: Any,
    device_buffer: Any,
    host_view: np.ndarray,
    stream: Any,
) -> None:
    """Issue an async host→device copy on ``stream``."""
    nbytes = int(host_view.nbytes)
    if nbytes == 0:
        return
    src_ptr = host_view.ctypes.data
    dst_ptr = int(device_buffer.data.ptr)
    cp.cuda.runtime.memcpyAsync(
        dst_ptr,
        src_ptr,
        nbytes,
        cp.cuda.runtime.memcpyHostToDevice,
        stream.ptr,
    )


def _parallel_workers() -> int:
    """Resolve the worker count for parallel BED range reads.

    Returns ``min(os.cpu_count(), 8)`` (clamped to >=1).
    """
    try:
        cpu = int(os.cpu_count() or 1)
    except Exception:  # noqa: BLE001
        cpu = 1
    return max(1, min(8, cpu))


def _parallel_pread_chunk(
    *,
    bed_path: Path,
    n_samples: int,
    n_variants: int,
    count_a1: bool,
    chunk_idx: np.ndarray,
    src_bpv: int,
    dst_buf: np.ndarray,
    dst_offset: int,
) -> None:
    """Worker: open own fd, indexed-coalesced gather one chunk into dst_buf.

    Each worker opens its own ``open_bed`` so the underlying ``os.preadv`` does
    not race on a shared file offset. The returned numpy gather payload is then
    memcpy'd into the shared destination at ``dst_offset``. This is the right
    primitive for the bitpacked cold-load indexed path: the coalescer in
    ``_pread_indexed_variant_payload`` keeps each chunk read mostly sequential
    while the per-worker fds let the kernel saturate the device.
    """
    chunk_reader = open_bed(
        path=bed_path,
        iid_count=n_samples,
        sid_count=n_variants,
        count_A1=count_a1,
    )
    try:
        if chunk_reader._bed_fd is not None:
            _fadvise_sequential(chunk_reader._bed_fd)
        chunk_payload = chunk_reader._pread_indexed_variant_payload(
            chunk_idx, bytes_per_variant=src_bpv,
        )
    finally:
        # No explicit close on open_bed; the fd is process-lived. We just drop
        # the reader reference here so the next loader-level fadvise on the
        # parent reader is independent.
        pass
    dst_buf[dst_offset : dst_offset + chunk_payload.nbytes] = chunk_payload


def _read_packed_parallel(
    bed_path: Path,
    *,
    payload_offset: int,
    total_bytes: int,
    pinned_view: np.ndarray,
    n_workers: int,
) -> None:
    """Range-read ``total_bytes`` from ``bed_path`` starting at ``payload_offset``
    using ``n_workers`` independent fds via ``os.pread``.

    Each worker handles a contiguous byte range and writes directly into the
    matching slice of ``pinned_view``. ``os.pread`` is thread-safe at the OS
    level and using separate fds avoids any kernel-side file-offset
    serialization quirks (gcsfuse + multi-threaded readv specifically).
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading as _threading
    import time as _time

    if total_bytes <= 0 or n_workers <= 0:
        return
    if int(pinned_view.nbytes) < int(total_bytes):
        raise ValueError(
            f"pinned_view too small: {int(pinned_view.nbytes)} < {int(total_bytes)} bytes"
        )

    chunk = (total_bytes + n_workers - 1) // n_workers

    try:
        from sv_pgs.progress import log as _log
    except ImportError:  # pragma: no cover
        _log = None  # type: ignore[assignment]

    # Aggregated progress counter shared across workers. itertools.count is a
    # C-level atomic increment so workers can bump it without holding a lock;
    # the emitter thread snapshots the value periodically.
    bytes_done = [0]
    bytes_done_lock = _threading.Lock()
    stop_event = _threading.Event()
    t_start = _time.monotonic()
    total_gb = total_bytes / 1e9

    def _emitter() -> None:
        if _log is None:
            return
        last_emitted = 0
        log_every_bytes = 256 * 1024 * 1024
        while not stop_event.wait(0.5):
            with bytes_done_lock:
                cur = bytes_done[0]
            if cur - last_emitted >= log_every_bytes:
                elapsed = max(_time.monotonic() - t_start, 1e-6)
                mb_per_sec = (cur / 1e6) / elapsed
                remaining = max(total_bytes - cur, 0)
                eta_sec = remaining / max(cur / elapsed, 1.0)
                _log(
                    f"BED stream: {cur / 1e9:.2f} / {total_gb:.2f} GB "
                    f"({mb_per_sec:.1f} MB/s, ETA {eta_sec / 60.0:.1f} min) "
                    f"[parallel x{n_workers}]"
                )
                last_emitted = cur

    # Open one fd per worker. Best-effort SEQUENTIAL hint per fd so gcsfuse can
    # issue large prefetches on each stream independently.
    fds: list[int] = []
    emitter_thread = _threading.Thread(target=_emitter, name="bed-stream-progress", daemon=True)
    try:
        for _ in range(n_workers):
            fd = os.open(str(bed_path), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            _fadvise_sequential(fd)
            fds.append(fd)

        def _worker(worker_idx: int) -> None:
            start = worker_idx * chunk
            stop = min(total_bytes, start + chunk)
            if start >= stop:
                return
            fd = fds[worker_idx]
            cursor = start
            # Read in sub-chunks of up to 64 MB so a single pread call doesn't
            # block forever on a slow stream.
            sub = 64 * 1024 * 1024
            while cursor < stop:
                want = min(sub, stop - cursor)
                buf = os.pread(fd, want, payload_offset + cursor)
                got = len(buf)
                if got == 0:
                    raise RuntimeError(
                        f"pread returned 0 bytes at offset {payload_offset + cursor} "
                        f"(worker={worker_idx}); BED truncated?"
                    )
                pinned_view[cursor : cursor + got] = np.frombuffer(buf, dtype=np.uint8)
                cursor += got
                with bytes_done_lock:
                    bytes_done[0] += got

        emitter_thread.start()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_worker, i) for i in range(n_workers)]
            for f in futures:
                f.result()
        # Final summary log so callers observing logs always see a terminal
        # `BED stream: ... DONE` line for the parallel path.
        if _log is not None:
            with bytes_done_lock:
                cur = bytes_done[0]
            elapsed = max(_time.monotonic() - t_start, 1e-6)
            mb_per_sec = (cur / 1e6) / elapsed
            _log(
                f"BED stream: {cur / 1e9:.2f} / {total_gb:.2f} GB "
                f"DONE ({mb_per_sec:.1f} MB/s, elapsed {elapsed / 60.0:.1f} min) "
                f"[parallel x{n_workers}]"
            )
    finally:
        stop_event.set()
        if emitter_thread.is_alive():
            try:
                emitter_thread.join(timeout=2.0)
            except RuntimeError:
                pass
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def _read_all_packed(
    reader: open_bed,
    n_variants: int,
    bytes_per_variant: int,
    pinned_view: np.ndarray,
    *,
    chunk_bytes: int = _FULL_READ_CHUNK_BYTES,
) -> None:
    """Stream all variants from the BED into ``pinned_view`` in ``chunk_bytes``
    chunks (default 256 MB)."""
    from sv_pgs.plink import PLINK1_HEADER_SIZE
    import time as _time
    try:
        from sv_pgs.progress import log as _log
    except ImportError:  # pragma: no cover - tests with stubbed progress module
        _log = None  # type: ignore[assignment]
    from sv_pgs.diagnostics import region as _diag_region, update_bytes as _diag_update

    total_bytes = n_variants * bytes_per_variant
    if total_bytes == 0:
        return
    chunk_variants = max(1, chunk_bytes // max(1, bytes_per_variant))
    cursor = 0
    bytes_read = 0
    last_log_bytes = 0
    log_every_bytes = 256 * 1024 * 1024  # 256 MB cadence per spec
    t_start = _time.monotonic()
    total_gb = total_bytes / 1e9
    _source_repr = str(getattr(reader, "filepath", None) or getattr(reader, "_path", "?"))
    with _diag_region("bed_stream", bytes_total=int(total_bytes), source=_source_repr):
     while cursor < n_variants:
        stop = min(n_variants, cursor + chunk_variants)
        byte_offset = PLINK1_HEADER_SIZE + cursor * bytes_per_variant
        byte_length = (stop - cursor) * bytes_per_variant
        payload = reader._pread_payload(byte_offset, byte_length)
        host_offset = cursor * bytes_per_variant
        pinned_view[host_offset : host_offset + byte_length] = payload
        cursor = stop
        bytes_read += byte_length
        _diag_update("bed_stream", bytes_read)
        if _log is not None and bytes_read - last_log_bytes >= log_every_bytes:
            elapsed = max(_time.monotonic() - t_start, 1e-6)
            mb_per_sec = (bytes_read / 1e6) / elapsed
            remaining = max(total_bytes - bytes_read, 0)
            eta_sec = remaining / max(bytes_read / elapsed, 1.0)
            _log(
                f"BED stream: {bytes_read / 1e9:.2f} / {total_gb:.2f} GB "
                f"({mb_per_sec:.1f} MB/s, ETA {eta_sec / 60.0:.1f} min)"
            )
            last_log_bytes = bytes_read
    if _log is not None and bytes_read > 0:
        elapsed = max(_time.monotonic() - t_start, 1e-6)
        mb_per_sec = (bytes_read / 1e6) / elapsed
        _log(
            f"BED stream: {bytes_read / 1e9:.2f} / {total_gb:.2f} GB "
            f"DONE ({mb_per_sec:.1f} MB/s, elapsed {elapsed / 60.0:.1f} min)"
        )


def _rebitpack_for_samples(
    payload: np.ndarray,
    *,
    n_variants: int,
    src_bytes_per_variant: int,
    src_n_samples: int,
    sample_indices: np.ndarray,
    count_a1: bool,
) -> tuple[np.ndarray, int]:
    """Decode → gather samples → re-encode using the plink.py LUTs.

    Returns ``(new_packed_flat, new_bytes_per_variant)``. ``new_packed_flat`` is
    a flat uint8 numpy array of length ``n_variants * new_bytes_per_variant`` so
    the caller can copy it into the pinned staging buffer directly.
    """
    if sample_indices.ndim != 1:
        raise ValueError("sample_indices must be 1D.")
    samples = np.ascontiguousarray(sample_indices, dtype=np.intp)
    if samples.size > 0 and (samples.min() < 0 or samples.max() >= src_n_samples):
        raise IndexError("sample_indices contains out-of-bounds entries.")

    new_n_samples = int(samples.size)
    new_bpv = _bytes_per_variant(new_n_samples)
    if new_n_samples == 0:
        return np.empty((0,), dtype=np.uint8), new_bpv

    lut = _BYTE_DECODE_LUT_A1 if count_a1 else _BYTE_DECODE_LUT_A2
    # PLINK_MISSING_INT8 sentinel (matches plink.py).
    missing_i8 = np.int8(-127)

    packed_matrix = payload.reshape(n_variants, src_bytes_per_variant)
    out = np.empty((n_variants, new_bpv), dtype=np.uint8)

    # Process variants in row chunks to keep the int8 decode buffer bounded.
    # 64 MB worth of decoded int8 per chunk: chunk_rows * src_bpv*4 ≤ 64 MB.
    decode_chunk_bytes = 64 * 1024 * 1024
    row_decoded_bytes = max(1, src_bytes_per_variant * 4)
    chunk_rows = max(1, decode_chunk_bytes // row_decoded_bytes)

    def _process_chunk(row_start: int, row_stop: int) -> None:
        packed_block = packed_matrix[row_start:row_stop, :]
        # Decode: (chunk_rows, src_bpv) -> (chunk_rows, src_bpv * 4) int8.
        decoded = lut[packed_block].reshape(row_stop - row_start, src_bytes_per_variant * 4)
        # Gather selected samples.
        gathered = decoded[:, : src_n_samples][:, samples]
        # Re-encode: replicate sv_pgs.plink._encode_variant inline, vectorized
        # across rows. Missing slots (sentinel) -> 0b01; observed slots use the
        # A1 encoding LUT mapping {0,1,2} -> {0b11, 0b10, 0b00}.
        codes = np.zeros((row_stop - row_start, new_bpv * 4), dtype=np.uint8)
        observed = gathered != missing_i8
        dosages = np.where(observed, gathered, 0).astype(np.intp, copy=False)
        encoded_obs = _ENCODE_LOOKUP_A1[dosages]
        codes[:, :new_n_samples] = np.where(
            observed,
            encoded_obs,
            np.uint8(0b01),
        )
        packed_rows = (
            codes[:, 0::4]
            | (codes[:, 1::4] << 2)
            | (codes[:, 2::4] << 4)
            | (codes[:, 3::4] << 6)
        ).astype(np.uint8, copy=False)
        out[row_start:row_stop, :] = packed_rows

    # Embarrassingly parallel over chunks: each chunk reads its own slice of
    # packed_matrix and writes its own slice of out. NumPy LUT/gather/encode
    # ops release the GIL, so ThreadPoolExecutor gives near-linear speedup
    # across cores (saw ~10x on 12-core AoU workbench).
    chunk_starts = list(range(0, n_variants, chunk_rows))
    n_workers = _parallel_workers()
    if n_workers <= 1 or len(chunk_starts) <= 1:
        for row_start in chunk_starts:
            row_stop = min(n_variants, row_start + chunk_rows)
            _process_chunk(row_start, row_stop)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_process_chunk, row_start, min(n_variants, row_start + chunk_rows))
                for row_start in chunk_starts
            ]
            for fut in futures:
                fut.result()

    return out.reshape(-1), new_bpv


def _compute_mean_std_via_screening(
    cp: Any,
    packed_device: Any,
    n_samples: int,
    count_a1: bool,
    stream: Any,
) -> tuple[Any, Any]:
    """Run the screening kernel and convert raw sums to (mean, std) float32."""
    from sv_pgs.bitpacked.screening import screen  # lazy import

    n_variants = int(packed_device.shape[0])
    out_count = cp.zeros((n_variants,), dtype=cp.int32)
    out_sum = cp.zeros((n_variants,), dtype=cp.float64)
    out_sumsq = cp.zeros((n_variants,), dtype=cp.float64)

    screen(
        packed_device,
        n_samples,
        out_count=out_count,
        out_sum=out_sum,
        out_sumsq=out_sumsq,
        y_resid=None,
        out_y_dot=None,
        count_a1=count_a1,
        stream=stream,
    )

    count_f64 = out_count.astype(cp.float64)
    safe_count = cp.where(count_f64 > 0, count_f64, cp.float64(1.0))
    mean64 = out_sum / safe_count
    var64 = out_sumsq / safe_count - mean64 * mean64
    var64 = cp.where(var64 > 0, var64, cp.float64(0.0))
    std64 = cp.sqrt(var64)

    has_obs = count_f64 > 0
    mean64 = cp.where(has_obs, mean64, cp.float64(0.0))
    # std==0 columns (constant variants) and count==0 columns are imputed to 1
    # so the standardization divides cleanly (mean is 0 for count==0 above).
    std64 = cp.where(has_obs & (std64 > 0), std64, cp.float64(1.0))

    return mean64.astype(cp.float32), std64.astype(cp.float32)


def load_bed_to_bitpacked_device(
    bed_path: str | Path,
    n_samples: int,
    n_variants: int,
    variant_indices: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    count_a1: bool = True,
    stream: Any | None = None,
) -> "BitpackedDeviceMatrix":
    """Read a PLINK 1.9 BED file into a device-resident ``BitpackedDeviceMatrix``.

    Parameters
    ----------
    bed_path
        Path to the ``.bed`` file (its ``.bim``/``.fam`` siblings are not read).
    n_samples
        Total sample count in the on-disk BED.
    n_variants
        Total variant count in the on-disk BED.
    variant_indices
        Optional 1D index array selecting a variant subset. When provided the
        coalesced indexed reader (``open_bed._pread_indexed_variant_payload``)
        is used so sparse-but-sorted gathers stay sequential.
    sample_indices
        Optional 1D index array selecting a sample subset. When provided the
        packed buffer is decoded, sample-gathered, and re-encoded on the CPU
        before the device upload — bitpacking cannot be sliced contiguously
        for an arbitrary permutation.
    mean, std
        Optional precomputed per-variant float32 mean / std vectors with
        length matching the post-gather variant count. When omitted both are
        computed on-device via the screening kernel after the upload.
    count_a1
        PLINK1 count-A1 convention (matches sv_pgs default).
    stream
        Optional CuPy stream. ``None`` uses the current stream.
    """
    cp = _require_cupy()

    bed_path = Path(bed_path)
    n_samples = int(n_samples)
    n_variants = int(n_variants)
    if n_samples < 1 or n_variants < 1:
        raise ValueError("n_samples and n_variants must be positive.")

    # The cold load is a single sequential sweep of the BED — accept a
    # gcsfuse-backed source here (forcing a 194 GB local copy first is worse
    # than streaming once at ~200 MB/s). assert_safe_for_purpose still rejects
    # gs:// URIs, missing paths and NFS/SSHFS mounts.
    assert_safe_for_purpose(
        bed_path,
        purpose="bitpacked_loader.load_bed_to_bitpacked_device",
        allow_sequential_gcsfuse=True,
    )
    try:
        source_on_gcsfuse = is_gcsfuse_path(bed_path)
    except Exception:
        source_on_gcsfuse = False
    if source_on_gcsfuse:
        _LOG.info(
            "load_bed_to_bitpacked_device: streaming directly from gcsfuse "
            "(sequential read, no local copy): %s",
            bed_path,
        )

    src_bpv = _bytes_per_variant(n_samples)

    # 1. Read packed bytes (all variants or coalesced gather).
    if variant_indices is None:
        gathered_n_variants = n_variants
        reader = open_bed(
            path=bed_path,
            iid_count=n_samples,
            sid_count=n_variants,
            count_A1=count_a1,
        )
        # For the one-shot cold load we want SEQUENTIAL read-ahead, which
        # overrides the RANDOM hint open_bed installs by default. On gcsfuse
        # this is what lets the kernel issue large prefetches; on local NVMe
        # it's a no-op-ish best-effort hint.
        if reader._bed_fd is not None:
            _fadvise_sequential(reader._bed_fd)
        # On gcsfuse, force the read chunk above _GCSFUSE_MIN_CHUNK_BYTES so
        # each HTTP GET to GCS is a fat request rather than a stream of small
        # ones. (Default 256 MB already exceeds the 16 MB floor.)
        chunk_bytes = (
            max(_FULL_READ_CHUNK_BYTES, _GCSFUSE_MIN_CHUNK_BYTES)
            if source_on_gcsfuse
            else _FULL_READ_CHUNK_BYTES
        )
        try:
            raw_total_bytes = gathered_n_variants * src_bpv
            # Read directly into pinned memory when no sample rebitpack is needed
            # to avoid an intermediate copy.
            if sample_indices is None:
                pinned_mem, pinned_view = _allocate_pinned(cp, raw_total_bytes)
                # Parallel range-read path: N independent fds, ``os.pread`` on
                # each. Used when the worker count resolves to >1 AND the BED
                # is gcsfuse-backed AND total payload exceeds the 1 GB
                # threshold. On local NVMe multiple workers just thrash
                # the page cache vs. a single sequential stream, so we skip.
                # Falls back to the sequential ``_read_all_packed`` path on any
                # error.
                n_workers = _parallel_workers()
                used_parallel = False
                parallel_eligible = (
                    n_workers > 1
                    and raw_total_bytes > _PARALLEL_GCSFUSE_THRESHOLD_BYTES
                    and source_on_gcsfuse
                )
                if parallel_eligible:
                    from sv_pgs.plink import PLINK1_HEADER_SIZE
                    try:
                        _read_packed_parallel(
                            bed_path,
                            payload_offset=PLINK1_HEADER_SIZE,
                            total_bytes=raw_total_bytes,
                            pinned_view=pinned_view,
                            n_workers=n_workers,
                        )
                        used_parallel = True
                        try:
                            from sv_pgs.progress import log as _plog
                            _plog(
                                f"BED stream: parallel range-read OK "
                                f"workers={n_workers} bytes={raw_total_bytes / 1e9:.2f} GB"
                            )
                        except ImportError:  # pragma: no cover
                            pass
                    except Exception as _parallel_exc:  # noqa: BLE001
                        # Best-effort fallback: log & retry via the sequential
                        # streaming path so we never break a run on a parallel
                        # I/O hiccup.
                        try:
                            from sv_pgs.progress import log as _plog
                            _plog(
                                f"BED stream: parallel range-read failed "
                                f"({type(_parallel_exc).__name__}: {_parallel_exc}); "
                                f"falling back to sequential"
                            )
                        except ImportError:  # pragma: no cover
                            pass
                if not used_parallel:
                    _read_all_packed(
                        reader,
                        gathered_n_variants,
                        src_bpv,
                        pinned_view,
                        chunk_bytes=chunk_bytes,
                    )
                packed_for_upload = pinned_view
                final_bpv = src_bpv
            else:
                # Read into a plain numpy buffer; rebitpack will produce the
                # final layout sized for the chosen samples.
                raw = np.empty((raw_total_bytes,), dtype=np.uint8)
                _read_all_packed(
                    reader,
                    gathered_n_variants,
                    src_bpv,
                    raw,
                    chunk_bytes=chunk_bytes,
                )
                packed_for_upload = raw  # placeholder, replaced below
                final_bpv = src_bpv
        finally:
            # open_bed currently has no explicit close; the fd is process-lived.
            # Hint POSIX_FADV_DONTNEED so the kernel can drop the BED pages from
            # the page cache — we hold the bytes in HBM and won't re-read them.
            if source_on_gcsfuse and reader._bed_fd is not None:
                _fadvise_dontneed(reader._bed_fd)
    else:
        var_idx = np.ascontiguousarray(np.asarray(variant_indices, dtype=np.int64))
        if var_idx.ndim != 1:
            raise ValueError("variant_indices must be 1D.")
        if var_idx.size > 0 and (var_idx.min() < 0 or var_idx.max() >= n_variants):
            raise IndexError("variant_indices contains out-of-bounds entries.")
        gathered_n_variants = int(var_idx.size)
        reader = open_bed(
            path=bed_path,
            iid_count=n_samples,
            sid_count=n_variants,
            count_A1=count_a1,
        )
        # Indexed gather is still mostly-sequential thanks to the coalescer,
        # and a one-shot cold load benefits from SEQUENTIAL read-ahead.
        if reader._bed_fd is not None:
            _fadvise_sequential(reader._bed_fd)
        # Parallel indexed pread: split var_idx into N contiguous variant-order
        # chunks, each worker opens its own fd and gathers its chunk via the
        # coalesced indexed pread. Local NVMe under overlayfs sustains ~40 MB/s
        # per-thread but ~5-8x that aggregate with parallel fds. Write each
        # chunk into the pre-allocated payload buffer at its byte offset.
        gather_workers = _parallel_workers()
        gathered_bytes_total = int(var_idx.size) * src_bpv
        payload = np.empty((gathered_bytes_total,), dtype=np.uint8)
        if gather_workers <= 1 or int(var_idx.size) < 2048:
            chunk_payload = reader._pread_indexed_variant_payload(
                var_idx, bytes_per_variant=src_bpv,
            )
            payload[:] = chunk_payload
            del chunk_payload
        else:
            chunk_size = (int(var_idx.size) + gather_workers - 1) // gather_workers
            futures = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=gather_workers
            ) as pool:
                for w in range(gather_workers):
                    start = w * chunk_size
                    if start >= int(var_idx.size):
                        break
                    stop = min(start + chunk_size, int(var_idx.size))
                    chunk_idx = var_idx[start:stop]
                    dst_offset = start * src_bpv
                    futures.append(
                        pool.submit(
                            _parallel_pread_chunk,
                            bed_path=bed_path,
                            n_samples=n_samples,
                            n_variants=n_variants,
                            count_a1=count_a1,
                            chunk_idx=chunk_idx,
                            src_bpv=src_bpv,
                            dst_buf=payload,
                            dst_offset=dst_offset,
                        )
                    )
                for fut in futures:
                    fut.result()
            try:
                from sv_pgs.progress import log as _plog
                _plog(
                    f"BED stream: parallel indexed gather OK "
                    f"workers={gather_workers} bytes={gathered_bytes_total / 1e9:.2f} GB"
                )
            except ImportError:  # pragma: no cover
                pass
        if sample_indices is None:
            # Move the gather result into a pinned staging buffer.
            pinned_mem, pinned_view = _allocate_pinned(cp, payload.nbytes)
            pinned_view[:] = payload
            packed_for_upload = pinned_view
            final_bpv = src_bpv
        else:
            packed_for_upload = payload
            final_bpv = src_bpv  # placeholder, replaced after rebitpack
        if source_on_gcsfuse and reader._bed_fd is not None:
            _fadvise_dontneed(reader._bed_fd)

    # 2. Optional CPU rebitpack onto the requested sample subset.
    if sample_indices is not None:
        s_idx = np.asarray(sample_indices, dtype=np.intp)
        # ``packed_for_upload`` here is a plain numpy array carrying every
        # source sample byte for ``gathered_n_variants`` variants.
        rebitpacked, final_bpv = _rebitpack_for_samples(
            np.ascontiguousarray(packed_for_upload),
            n_variants=gathered_n_variants,
            src_bytes_per_variant=src_bpv,
            src_n_samples=n_samples,
            sample_indices=s_idx,
            count_a1=count_a1,
        )
        effective_n_samples = int(s_idx.size)
        pinned_mem, pinned_view = _allocate_pinned(cp, rebitpacked.nbytes)
        pinned_view[:] = rebitpacked
        packed_for_upload = pinned_view
    else:
        effective_n_samples = n_samples

    # 3. Allocate device buffer and async DMA from pinned host memory.
    cp_stream = _resolve_stream(cp, stream)
    device_packed = cp.empty((gathered_n_variants, final_bpv), dtype=cp.uint8)
    _h2d_async(cp, device_packed, packed_for_upload, cp_stream)

    # 4. Resolve mean/std — either trust caller values or compute on device.
    if mean is not None and std is not None:
        mean_arr = np.ascontiguousarray(np.asarray(mean, dtype=np.float32))
        std_arr = np.ascontiguousarray(np.asarray(std, dtype=np.float32))
        if mean_arr.shape != (gathered_n_variants,):
            raise ValueError(
                f"mean must have shape ({gathered_n_variants},), got {mean_arr.shape}"
            )
        if std_arr.shape != (gathered_n_variants,):
            raise ValueError(
                f"std must have shape ({gathered_n_variants},), got {std_arr.shape}"
            )
        # Replace zero-std columns with 1.0 (matches preprocessing convention).
        std_arr = np.where(std_arr > 0, std_arr, np.float32(1.0))
        device_mean = cp.asarray(mean_arr)
        device_std = cp.asarray(std_arr)
        cp_stream.synchronize()
    elif mean is None and std is None:
        # Need the upload to land before screening reads the packed buffer.
        cp_stream.synchronize()
        device_mean, device_std = _compute_mean_std_via_screening(
            cp,
            device_packed,
            effective_n_samples,
            count_a1,
            cp_stream,
        )
    else:
        raise ValueError("mean and std must both be provided or both be None.")

    # 5. Wrap and return. Import lazily so this module imports cleanly even
    # when the matrix module is built later in the wave.
    from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix  # lazy

    return BitpackedDeviceMatrix(
        packed=device_packed,
        mean=device_mean,
        std=device_std,
        n_samples=effective_n_samples,
        count_a1=count_a1,
    )


# ---------------------------------------------------------------------------
# Persistent active-matrix cache
# ---------------------------------------------------------------------------

_ACTIVE_CACHE_SCHEMA_VERSION = 1


def _active_cache_content_hash(
    *,
    bed_path: Path,
    sample_indices: np.ndarray | None,
    variant_indices: np.ndarray | None,
    count_a1: bool,
) -> str:
    """Stable sha256 over the inputs that uniquely determine the cached matrix.

    Includes bed_path stat (size + mtime), sample/variant index bytes, and the
    count_a1 flag. We deliberately stat the BED rather than CRC the contents —
    194 GB of CRC at cache-key time would defeat the purpose.
    """
    h = hashlib.sha256()
    h.update(f"schema={_ACTIVE_CACHE_SCHEMA_VERSION}\n".encode("utf-8"))
    h.update(f"count_a1={int(bool(count_a1))}\n".encode("utf-8"))
    try:
        st = bed_path.stat()
        h.update(f"bed_size={st.st_size}\n".encode("utf-8"))
        # ns precision so a one-second overwrite is still detected.
        h.update(f"bed_mtime_ns={int(st.st_mtime_ns)}\n".encode("utf-8"))
    except OSError:
        h.update(f"bed_unstattable={bed_path!s}\n".encode("utf-8"))
    if sample_indices is None:
        h.update(b"sample_indices=None\n")
    else:
        s = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64))
        h.update(b"sample_indices=")
        h.update(s.tobytes())
        h.update(b"\n")
    if variant_indices is None:
        h.update(b"variant_indices=None\n")
    else:
        v = np.ascontiguousarray(np.asarray(variant_indices, dtype=np.int64))
        h.update(b"variant_indices=")
        h.update(v.tobytes())
        h.update(b"\n")
    return h.hexdigest()


def _active_cache_dir(root: Path, content_hash: str) -> Path:
    return Path(root) / "bitpacked_active" / content_hash


def _active_cache_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def verify_active_matrix_cache(cache_dir: Path) -> bool:
    """Return True iff ``cache_dir`` contains a complete, well-formed cache."""
    cache_dir = Path(cache_dir)
    manifest_path = _active_cache_manifest_path(cache_dir)
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, "rb") as fh:
            manifest = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    try:
        if int(manifest.get("schema_version", -1)) != _ACTIVE_CACHE_SCHEMA_VERSION:
            return False
        if not bool(manifest.get("complete", False)):
            return False
        expected_packed = int(manifest.get("packed_bytes", -1))
        n_samples = int(manifest.get("n_samples", -1))
        n_variants = int(manifest.get("n_variants", -1))
    except (TypeError, ValueError):
        return False
    if n_samples < 0 or n_variants < 0:
        return False
    packed = cache_dir / "packed.bin"
    mean = cache_dir / "mean.npy"
    scale = cache_dir / "scale.npy"
    nsf = cache_dir / "n_samples.txt"
    for required in (packed, mean, scale, nsf):
        if not required.exists():
            return False
    try:
        if packed.stat().st_size != expected_packed:
            return False
    except OSError:
        return False
    return True


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_active_matrix_cache(
    cache_dir: Path,
    *,
    matrix: "BitpackedDeviceMatrix",
    bed_path: Path,
    content_hash: str,
    count_a1: bool,
) -> None:
    """Atomically dump ``matrix`` (packed/mean/scale + n_samples + manifest).

    Each large array is written to ``<name>.partial.<pid>.<uuid>`` then
    fsync+renamed; the manifest with ``complete=true`` is published LAST so a
    crash mid-write leaves a clearly-incomplete cache (manifest absent).
    """
    cp = _require_cupy()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".partial.{os.getpid()}.{uuid.uuid4().hex[:8]}"

    packed_host = cp.asnumpy(matrix._packed)
    mean_host = cp.asnumpy(matrix._mean)
    std_host = cp.asnumpy(matrix._std)

    packed_partial = cache_dir / f"packed.bin{suffix}"
    mean_partial = cache_dir / f"mean.npy{suffix}"
    scale_partial = cache_dir / f"scale.npy{suffix}"
    ns_partial = cache_dir / f"n_samples.txt{suffix}"

    try:
        with open(packed_partial, "wb") as fh:
            fh.write(np.ascontiguousarray(packed_host).tobytes())
            fh.flush()
            os.fsync(fh.fileno())
        # np.save appends ``.npy`` to a path lacking that extension; the
        # ``.partial.<pid>.<uuid>`` suffix means we MUST disable that fixup or
        # the actual on-disk filename diverges from what we rename. Use the
        # ``arr.tofile``-equivalent via the lower-level numpy.format.
        with open(mean_partial, "wb") as fh:
            np.lib.format.write_array(
                fh, np.ascontiguousarray(mean_host.astype(np.float32, copy=False))
            )
            fh.flush()
            os.fsync(fh.fileno())
        with open(scale_partial, "wb") as fh:
            np.lib.format.write_array(
                fh, np.ascontiguousarray(std_host.astype(np.float32, copy=False))
            )
            fh.flush()
            os.fsync(fh.fileno())
        with open(ns_partial, "w", encoding="utf-8") as fh:
            fh.write(str(int(matrix._n_samples)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(str(packed_partial), str(cache_dir / "packed.bin"))
        os.replace(str(mean_partial), str(cache_dir / "mean.npy"))
        os.replace(str(scale_partial), str(cache_dir / "scale.npy"))
        os.replace(str(ns_partial), str(cache_dir / "n_samples.txt"))
        _fsync_dir(cache_dir)

        manifest = {
            "schema_version": _ACTIVE_CACHE_SCHEMA_VERSION,
            "content_hash": content_hash,
            "source_bed_path": str(bed_path),
            "n_samples": int(matrix._n_samples),
            "n_variants": int(matrix._n_variants),
            "packed_bytes": int(packed_host.nbytes),
            "count_a1": bool(count_a1),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True,
        }
        manifest_partial = cache_dir / f"manifest.json{suffix}"
        with open(manifest_partial, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(manifest_partial), str(_active_cache_manifest_path(cache_dir)))
        _fsync_dir(cache_dir)
    finally:
        for p in (packed_partial, mean_partial, scale_partial, ns_partial):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def _load_active_matrix_cache(
    cache_dir: Path,
    *,
    count_a1: bool,
) -> "BitpackedDeviceMatrix":
    """Load a verified cache directory into a fresh BitpackedDeviceMatrix.

    Uses ``np.memmap`` for the packed buffer so the host-resident copy is
    page-cache-backed; the device upload then goes through a pinned-pageable
    DMA staging buffer (same as the cold load path).
    """
    cp = _require_cupy()
    cache_dir = Path(cache_dir)
    with open(cache_dir / "n_samples.txt", "r", encoding="utf-8") as fh:
        n_samples = int(fh.read().strip())
    mean_host = np.load(cache_dir / "mean.npy")
    std_host = np.load(cache_dir / "scale.npy")
    n_variants = int(mean_host.shape[0])
    bpv = _bytes_per_variant(n_samples)
    packed_size = n_variants * bpv
    packed_host = np.memmap(
        cache_dir / "packed.bin",
        dtype=np.uint8,
        mode="r",
        shape=(packed_size,),
    )
    # Stage into pinned memory for a fast async H2D upload.
    pinned_mem, pinned_view = _allocate_pinned(cp, packed_size)
    pinned_view[:] = packed_host
    del packed_host
    device_packed = cp.empty((n_variants, bpv), dtype=cp.uint8)
    cp_stream = _resolve_stream(cp, None)
    _h2d_async(cp, device_packed, pinned_view, cp_stream)
    device_mean = cp.asarray(mean_host.astype(np.float32, copy=False))
    device_std = cp.asarray(std_host.astype(np.float32, copy=False))
    cp_stream.synchronize()

    from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix  # lazy

    return BitpackedDeviceMatrix(
        packed=device_packed,
        mean=device_mean,
        std=device_std,
        n_samples=n_samples,
        count_a1=count_a1,
    )


def load_bed_to_bitpacked_device_cached(
    bed_path: str | Path,
    n_samples: int,
    n_variants: int,
    *,
    cache_dir: str | Path,
    variant_indices: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    count_a1: bool = True,
    stream: Any | None = None,
) -> "BitpackedDeviceMatrix":
    """Cache-aware wrapper around :func:`load_bed_to_bitpacked_device`.

    On cache HIT, mmaps ``<cache_dir>/bitpacked_active/<hash>/packed.bin`` and
    uploads to HBM (≈3-5s for a 24 GB matrix on local NVMe). On cache MISS,
    streams the BED and writes the cache atomically after the matrix is built.

    The cache is keyed by the sha256 of (bed_path stat, sample_indices bytes,
    variant_indices bytes, count_a1). Bumping any of those forces a re-load.
    """
    bed_path = Path(bed_path)
    content_hash = _active_cache_content_hash(
        bed_path=bed_path,
        sample_indices=sample_indices,
        variant_indices=variant_indices,
        count_a1=count_a1,
    )
    cache_subdir = _active_cache_dir(Path(cache_dir), content_hash)
    try:
        from sv_pgs.progress import log as _log
    except ImportError:  # pragma: no cover - progress should always exist
        _log = None  # type: ignore[assignment]

    if verify_active_matrix_cache(cache_subdir):
        if _log is not None:
            _log(
                f"bitpacked active-matrix cache HIT: {cache_subdir} "
                f"(content_hash={content_hash[:12]}...)"
            )
        t0 = time.monotonic()
        matrix = _load_active_matrix_cache(cache_subdir, count_a1=count_a1)
        elapsed = time.monotonic() - t0
        if _log is not None:
            _log(
                f"bitpacked active-matrix cache load complete in {elapsed:.2f}s "
                f"(n_samples={matrix._n_samples}, n_variants={matrix._n_variants})"
            )
            try:
                resident_bytes = (
                    int(matrix._packed.nbytes)
                    + int(matrix._mean.nbytes)
                    + int(matrix._std.nbytes)
                )
                resident_gb = resident_bytes / 1e9
                _log(
                    f"BitpackedDeviceMatrix: loaded from cache {cache_subdir} "
                    f"({resident_gb:.1f} GB resident, {elapsed:.1f} sec)"
                )
            except Exception as _exc:  # noqa: BLE001 - logging never blocks
                _log(
                    f"BitpackedDeviceMatrix: loaded from cache {cache_subdir} "
                    f"({elapsed:.1f} sec; size log skipped: {_exc!r})"
                )
        return matrix

    if _log is not None:
        _log(
            f"bitpacked active-matrix cache MISS: streaming BED then populating "
            f"{cache_subdir} (content_hash={content_hash[:12]}...)"
        )
    # Sweep stale .partial.<pid>.<uuid> files from a prior killed run in this
    # cache subdir before re-attempting. The bitpacked active matrix is fully
    # reconstructable from the BED + indices, so any leftover partials are
    # garbage by definition (the manifest above already determined we missed).
    try:
        cache_subdir.mkdir(parents=True, exist_ok=True)
        for stale in cache_subdir.glob("*.partial.*"):
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass
    matrix = load_bed_to_bitpacked_device(
        bed_path=bed_path,
        n_samples=n_samples,
        n_variants=n_variants,
        variant_indices=variant_indices,
        sample_indices=sample_indices,
        mean=mean,
        std=std,
        count_a1=count_a1,
        stream=stream,
    )
    try:
        _write_active_matrix_cache(
            cache_subdir,
            matrix=matrix,
            bed_path=bed_path,
            content_hash=content_hash,
            count_a1=count_a1,
        )
        if _log is not None:
            _log(f"bitpacked active-matrix cache populated: {cache_subdir}")
    except Exception as exc:  # noqa: BLE001 - cache write is best-effort
        if _log is not None:
            _log(f"bitpacked active-matrix cache write failed (continuing): {exc!r}")
    return matrix

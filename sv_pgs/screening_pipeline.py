"""End-to-end NVMe -> screening pipeline.

Streams PLINK 1.9 BED shards in fixed-size byte chunks from disk into GPU
HBM, then runs the fused bitpacked screening kernel
(:func:`sv_pgs.bitpacked.screening.screen`) over each chunk to accumulate
per-variant ``count``, ``sum``, ``sumsq`` and (optionally) ``y_dot``.

The pipeline is structured as a classic two-buffer producer/consumer:
two pinned host-side staging buffers and two device-side staging buffers
are pre-allocated. While buffer ``i`` is being filled from NVMe + DMA'd
to the GPU on the "copy" CuPy stream, buffer ``1 - i`` has its screening
kernel launched on the "compute" stream. Cross-stream synchronization
uses CUDA events so that NVMe reads, host->device DMA, and kernel
execution overlap end-to-end.

When :func:`sv_pgs.gds.gpudirect_available` reports True, the host pinned
buffers are skipped and ``cufile_read_to_device`` reads directly into the
device staging buffers, eliminating the host bounce entirely.

This module performs only lazy imports of CuPy / GDS / the bitpacked
kernel package, so it imports cleanly in CPU-only environments.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from sv_pgs.plink import (
    PLINK1_HEADER_SIZE,
    PLINK1_MAGIC,
    _bytes_per_variant,
)

_LOG = logging.getLogger(__name__)

# Default streaming chunk size in bytes. ~256 MiB matches NVMe Gen4 burst
# sizes well and keeps a single staging buffer comfortably small relative
# to T4 / A100 HBM. The actual chunk size is rounded down to a whole
# number of variant rows to keep the screening kernel's row-per-block
# layout simple.
_DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Sample-intersect rebitpack (CPU path)
# ---------------------------------------------------------------------------


def _rebitpack_chunk(
    packed_chunk: np.ndarray,
    n_samples_src: int,
    sample_intersect: np.ndarray,
    count_a1: bool,
) -> np.ndarray:
    """Decode -> gather -> re-encode a packed chunk to a new sample stride.

    Parameters
    ----------
    packed_chunk : np.ndarray (n_variants_in_chunk, bytes_per_variant_src) uint8
    n_samples_src : original sample count.
    sample_intersect : int indices into the source samples, length n_samples_out.
    count_a1 : PLINK count-A1 convention (must match the source file's encoding).

    Returns
    -------
    np.ndarray (n_variants_in_chunk, (n_out + 3) // 4) uint8
    """
    # Lazy LUT import to avoid pulling cupy through bitpacked.__init__.
    from sv_pgs.plink import _BYTE_DECODE_LUT_A1, _BYTE_DECODE_LUT_A2

    lut = _BYTE_DECODE_LUT_A1 if count_a1 else _BYTE_DECODE_LUT_A2
    n_variants_chunk, bpv_src = packed_chunk.shape
    # Decode to (n_variants_chunk, 4 * bpv_src) int8, then trim to n_samples_src,
    # gather sample_intersect, and re-pack.
    decoded_full = lut[packed_chunk]  # (n_variants_chunk, bpv_src, 4) int8
    decoded = decoded_full.reshape(n_variants_chunk, -1)[:, :n_samples_src]
    gathered = decoded[:, sample_intersect]  # int8

    n_out = int(sample_intersect.shape[0])
    bpv_out = (n_out + 3) // 4
    padded = np.zeros((n_variants_chunk, bpv_out * 4), dtype=np.int8)
    padded[:, :n_out] = gathered

    # Encode each int8 dosage back to its 2-bit code (inverse of LUT).
    # Missing slots in plink.py are PLINK_MISSING_INT8 == -127.
    # Codes (count_a1=True):  2->0b00, miss->0b01, 1->0b10, 0->0b11
    # Codes (count_a1=False): 0->0b00, miss->0b01, 1->0b10, 2->0b11
    if count_a1:
        code_map = {2: 0b00, -127: 0b01, 1: 0b10, 0: 0b11}
    else:
        code_map = {0: 0b00, -127: 0b01, 1: 0b10, 2: 0b11}
    codes = np.full(padded.shape, 0b01, dtype=np.uint8)  # default missing
    for dosage, code in code_map.items():
        codes[padded == dosage] = code

    # Pack 4 codes per byte, low-bit-first: byte = (s3<<6)|(s2<<4)|(s1<<2)|s0
    codes = codes.reshape(n_variants_chunk, bpv_out, 4)
    out = (
        codes[:, :, 0]
        | (codes[:, :, 1] << 2)
        | (codes[:, :, 2] << 4)
        | (codes[:, :, 3] << 6)
    ).astype(np.uint8)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# NVMe chunk iterator (host side)
# ---------------------------------------------------------------------------


def _iter_chunks(
    bed_path: Path,
    n_samples: int,
    n_variants: int,
    chunk_bytes: int,
) -> Iterator[tuple[int, int]]:
    """Yield (variant_start, variant_stop) pairs covering [0, n_variants)."""
    bpv = _bytes_per_variant(n_samples)
    if bpv == 0 or n_variants == 0:
        return
    variants_per_chunk = max(1, chunk_bytes // bpv)
    start = 0
    while start < n_variants:
        stop = min(n_variants, start + variants_per_chunk)
        yield start, stop
        start = stop


def _read_chunk_to_host(
    fh: Any,
    n_samples: int,
    variant_start: int,
    variant_stop: int,
    host_buffer: np.ndarray,
) -> np.ndarray:
    """Read a contiguous variant-range from an open BED file into host_buffer.

    Returns a view of host_buffer shaped (n_variants_in_chunk, bpv).
    """
    bpv = _bytes_per_variant(n_samples)
    n_rows = variant_stop - variant_start
    n_bytes = n_rows * bpv
    offset = PLINK1_HEADER_SIZE + variant_start * bpv
    fh.seek(offset)
    view = host_buffer[:n_bytes]
    mv = memoryview(view).cast("B")
    fh.readinto(mv)
    return view[:n_bytes].view(np.uint8).reshape(n_rows, bpv)


# ---------------------------------------------------------------------------
# Per-path streaming screening
# ---------------------------------------------------------------------------


def _verify_magic(bed_path: Path) -> None:
    with open(bed_path, "rb") as fh:
        magic = fh.read(PLINK1_HEADER_SIZE)
    if magic != PLINK1_MAGIC:
        raise ValueError(
            f"{bed_path}: bad PLINK1 magic; got {magic.hex(' ')}, expected "
            f"{PLINK1_MAGIC.hex(' ')}"
        )


def _screen_one_path(
    bed_path: Path,
    n_samples: int,
    n_variants: int,
    *,
    sample_intersect: np.ndarray | None,
    rhs_dev: Any | None,
    count_a1: bool,
    stream: Any | None,
    chunk_bytes: int,
) -> dict[str, Any]:
    """Stream one BED file through the GPU screening kernel.

    Returns a dict of cupy arrays (count, sum, sumsq, dosage_rhs,
    observed_rhs) of length ``n_variants`` (with a trailing k-axis when
    ``rhs_dev`` is 2-D). The caller concatenates across paths.

    Standardized inner product reconstruction (per-variant, per-rhs-column):

        Z_v . rhs[:, j] = (dosage_rhs[v, j] - mean[v] * observed_rhs[v, j]) / scale[v]
    """
    import cupy as cp  # lazy

    from sv_pgs.bitpacked.screening import screen

    _verify_magic(bed_path)

    # Sample stride that actually hits the GPU kernel.
    if sample_intersect is None:
        n_samples_eff = n_samples
    else:
        n_samples_eff = int(sample_intersect.shape[0])
    bpv_src = _bytes_per_variant(n_samples)
    bpv_eff = _bytes_per_variant(n_samples_eff)

    # Try the GPUDirect Storage fast path; fall back to pinned-host staging.
    use_gds = False
    cufile_read_to_device = None
    if sample_intersect is None:
        try:
            from sv_pgs.gds import cufile_read_to_device as _cufile, gpudirect_available

            if gpudirect_available():
                use_gds = True
                cufile_read_to_device = _cufile
        except Exception:
            use_gds = False

    # Pre-allocate two host pinned buffers + two device staging buffers
    # (sized to the largest possible chunk we will read). When GDS is in
    # use the host buffers are unused — we still allocate small ones to
    # keep the control flow uniform.
    chunks = list(_iter_chunks(bed_path, n_samples, n_variants, chunk_bytes))
    # rhs shape bookkeeping: kernel supports 1D (k=1) and 2D (k>1).
    if rhs_dev is not None:
        rhs_shape_tail: tuple[int, ...] = (
            () if rhs_dev.ndim == 1 else (int(rhs_dev.shape[1]),)
        )
    else:
        rhs_shape_tail = ()

    if not chunks:
        empty_drhs = (
            cp.zeros((0,) + rhs_shape_tail, dtype=cp.float64)
            if rhs_dev is not None
            else None
        )
        empty_orhs = (
            cp.zeros((0,) + rhs_shape_tail, dtype=cp.float64)
            if rhs_dev is not None
            else None
        )
        return {
            "count": cp.zeros(0, dtype=cp.int32),
            "sum": cp.zeros(0, dtype=cp.float64),
            "sumsq": cp.zeros(0, dtype=cp.float64),
            "dosage_rhs": empty_drhs,
            "observed_rhs": empty_orhs,
        }

    max_rows = max(stop - start for start, stop in chunks)
    host_byte_capacity = max_rows * bpv_src
    dev_byte_capacity = max_rows * bpv_eff

    host_bufs: list[np.ndarray] = []
    for _ in range(2):
        if use_gds:
            host_bufs.append(np.empty(0, dtype=np.uint8))
        else:
            host_bufs.append(_alloc_pinned_host(host_byte_capacity))

    # Allocate device staging via the standard CuPy device allocator. We
    # rebuild the buffers in correct effective stride (post-intersect) so
    # the screening kernel sees the right ``bytes_per_variant``.
    dev_bufs = [cp.empty(dev_byte_capacity, dtype=cp.uint8) for _ in range(2)]

    # Outputs (per-path).
    out_count = cp.zeros(n_variants, dtype=cp.int32)
    out_sum = cp.zeros(n_variants, dtype=cp.float64)
    out_sumsq = cp.zeros(n_variants, dtype=cp.float64)
    if rhs_dev is not None:
        out_dosage_rhs = cp.zeros((n_variants,) + rhs_shape_tail, dtype=cp.float64)
        out_observed_rhs = cp.zeros((n_variants,) + rhs_shape_tail, dtype=cp.float64)
    else:
        out_dosage_rhs = None
        out_observed_rhs = None

    # Two streams: copy (NVMe -> device) and compute (kernel).
    copy_stream = cp.cuda.Stream(non_blocking=True)
    compute_stream = stream if stream is not None else cp.cuda.Stream(non_blocking=True)
    copy_done = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
    compute_done = [cp.cuda.Event(disable_timing=True) for _ in range(2)]

    # Prefer mmap-based reads on local files; gcsfuse paths warn and use
    # the direct-open fallback. ``fh`` is kept as the fallback path for any
    # read error encountered while using the mmap reader.
    fh = None
    mmap_reader: Any = None
    if not use_gds:
        try:
            from sv_pgs.gcsfuse_staging import open_for_sequential_read

            _, is_local = open_for_sequential_read(Path(bed_path))
        except Exception:
            is_local = True
        if not is_local:
            _LOG.warning(
                "screening_pipeline: %s appears to be on gcsfuse; consider "
                "staging locally via sv_pgs.gcsfuse_staging.stage_to_local "
                "before screening for vastly faster sequential reads.",
                bed_path,
            )
        else:
            try:
                from sv_pgs.mmap_reader import BedMmapReader

                mmap_reader = BedMmapReader(
                    bed_path,
                    n_samples=n_samples,
                    n_variants=n_variants,
                    count_a1=count_a1,
                )
            except Exception as exc:
                _LOG.warning(
                    "screening_pipeline: BedMmapReader init failed for %s "
                    "(%s: %s); falling back to direct open.",
                    bed_path,
                    type(exc).__name__,
                    exc,
                )
                mmap_reader = None
        fh = open(bed_path, "rb", buffering=0)
        # On local-hot files, hint POSIX_FADV_SEQUENTIAL so the kernel pre-
        # reads the next chunk while we're DMAing the current one to HBM.
        # No-op on gcsfuse (the warning above already told the user to stage
        # locally). Best-effort: ignored where fadvise is unsupported.
        if is_local:
            _fadvise = getattr(os, "posix_fadvise", None)
            _seq = getattr(os, "POSIX_FADV_SEQUENTIAL", None)
            if _fadvise is not None and _seq is not None:
                try:
                    _fadvise(fh.fileno(), 0, 0, _seq)
                except OSError:
                    pass

    try:
        for idx, (v_start, v_stop) in enumerate(chunks):
            slot = idx % 2
            n_rows = v_stop - v_start
            n_bytes_src = n_rows * bpv_src
            n_bytes_eff = n_rows * bpv_eff
            offset = PLINK1_HEADER_SIZE + v_start * bpv_src

            # Make sure the previous iteration's compute on this slot has
            # finished before we overwrite the device staging buffer.
            if idx >= 2:
                copy_stream.wait_event(compute_done[slot])

            dev_slice = dev_bufs[slot][:n_bytes_eff]

            if use_gds and sample_intersect is None and cufile_read_to_device is not None:
                # Direct NVMe -> device. Still issued on copy_stream so the
                # subsequent kernel launch can wait on it via an event.
                with copy_stream:
                    cufile_read_to_device(bed_path, dev_slice, offset, n_bytes_src)
                copy_done[slot].record(copy_stream)
            else:
                # NVMe -> pinned host -> device. Prefer the mmap reader on
                # local files; fall back to the direct-open path on any
                # read error.
                packed_host = None
                if mmap_reader is not None:
                    try:
                        mmap_view = mmap_reader.read_packed_range(v_start, v_stop)
                        # Copy into the pinned host buffer so the existing
                        # downstream code (which expects a writable, packed
                        # flat view) works unchanged.
                        host_view = host_bufs[slot][:n_bytes_src].reshape(
                            n_rows, bpv_src
                        )
                        np.copyto(host_view, mmap_view)
                        packed_host = host_view
                    except Exception as exc:
                        _LOG.warning(
                            "screening_pipeline: mmap read failed for %s "
                            "[%d:%d] (%s: %s); falling back to direct open.",
                            bed_path,
                            v_start,
                            v_stop,
                            type(exc).__name__,
                            exc,
                        )
                        packed_host = None
                if packed_host is None:
                    packed_host = _read_chunk_to_host(
                        fh, n_samples, v_start, v_stop, host_bufs[slot]
                    )
                if sample_intersect is not None:
                    rebitpacked = _rebitpack_chunk(
                        packed_host, n_samples, sample_intersect, count_a1
                    )
                    src_bytes = rebitpacked.reshape(-1).view(np.uint8)
                else:
                    src_bytes = packed_host.reshape(-1).view(np.uint8)
                assert src_bytes.shape[0] == n_bytes_eff
                with copy_stream:
                    dev_slice.set(src_bytes, stream=copy_stream)
                copy_done[slot].record(copy_stream)

            # Kernel launches on compute stream after the DMA finishes.
            compute_stream.wait_event(copy_done[slot])
            packed_dev = dev_slice.reshape(n_rows, bpv_eff)
            screen(
                packed_dev,
                n_samples_eff,
                out_count[v_start:v_stop],
                out_sum[v_start:v_stop],
                out_sumsq[v_start:v_stop],
                rhs=rhs_dev,
                out_dosage_rhs=(
                    None if out_dosage_rhs is None else out_dosage_rhs[v_start:v_stop]
                ),
                out_observed_rhs=(
                    None if out_observed_rhs is None else out_observed_rhs[v_start:v_stop]
                ),
                count_a1=count_a1,
                stream=compute_stream,
            )
            compute_done[slot].record(compute_stream)
    finally:
        if fh is not None:
            fh.close()
        if mmap_reader is not None:
            try:
                mmap_reader.close()
            except Exception:
                pass

    # Drain.
    compute_stream.synchronize()
    copy_stream.synchronize()

    return {
        "count": out_count,
        "sum": out_sum,
        "sumsq": out_sumsq,
        "dosage_rhs": out_dosage_rhs,
        "observed_rhs": out_observed_rhs,
    }


def _alloc_pinned_host(n_bytes: int) -> np.ndarray:
    """Allocate a pinned host buffer as a numpy uint8 view of size ``n_bytes``."""
    import cupy as cp  # lazy

    mem = cp.cuda.alloc_pinned_memory(n_bytes)
    # cp.cuda.PinnedMemory exposes a buffer-protocol-compatible memoryview.
    arr = np.frombuffer(mem, dtype=np.uint8, count=n_bytes)
    return arr


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_screening_pass(
    bed_paths: list[Path] | list[str],
    n_samples_per_path: list[int],
    n_variants_per_path: list[int],
    sample_intersect: np.ndarray | None = None,
    rhs: np.ndarray | None = None,
    count_a1: bool = True,
    stream: Any | None = None,
    y_resid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Stream every BED shard through the bitpacked screening kernel.

    Parameters
    ----------
    bed_paths
        One or more PLINK 1.9 ``.bed`` paths. Each shard contributes a
        contiguous range of variants to the output (in input order).
    n_samples_per_path, n_variants_per_path
        Per-shard sample and variant counts. ``len(...)`` must equal
        ``len(bed_paths)``.
    sample_intersect
        Optional integer indices into each shard's samples (the same
        intersect must apply to every shard — this is the cohort-level
        sample subset). If given, every chunk is decoded -> gathered ->
        re-encoded on the CPU before upload.
    rhs
        Optional length-``n_samples_out`` float vector (1-D) or
        ``(n_samples_out, k)`` matrix (2-D). If given, the screening
        kernel additionally computes the missingness-aware partials
        ``dosage_rhs[v, j] = sum_{i in obs} d_iv * rhs[i, j]`` and
        ``observed_rhs[v, j] = sum_{i in obs} rhs[i, j]`` per
        variant. Uploaded to the device once and reused across shards.
        The standardized inner product is reconstructed via
        :func:`finalize_standardized_rhs`.
    count_a1
        PLINK count-A1 convention (default True, matches sv_pgs).
    stream
        Optional caller-provided CuPy compute stream. If None, a new
        non-blocking stream is created per shard.
    y_resid
        Deprecated alias for ``rhs``. Emits ``DeprecationWarning`` and
        will be removed in a future release. Mirrors the kernel-level
        deprecation in :mod:`sv_pgs.bitpacked.screening`.

    Returns
    -------
    dict with concatenated per-variant numpy arrays:
        ``count`` int32 (total_variants,)
        ``sum``   float64 (total_variants,)
        ``sumsq`` float64 (total_variants,)
        ``dosage_rhs``   float64 (total_variants, [k]) or None
        ``observed_rhs`` float64 (total_variants, [k]) or None

    To obtain the standardized inner product ``Z_v . y``, combine via::

        Z_v . y = (dosage_rhs - mean * observed_rhs) / scale

    See :func:`finalize_standardized_rhs`.
    """
    if len(bed_paths) != len(n_samples_per_path) or len(bed_paths) != len(
        n_variants_per_path
    ):
        raise ValueError(
            "bed_paths, n_samples_per_path, n_variants_per_path must have equal length"
        )

    import cupy as cp  # lazy

    # Back-compat: accept the deprecated ``y_resid=`` alias. Mirrors the
    # kernel-level deprecation in sv_pgs.bitpacked.screening.
    if y_resid is not None:
        if rhs is not None:
            raise TypeError(
                "run_screening_pass() received both `rhs` and the deprecated "
                "alias `y_resid`; pass only `rhs`."
            )
        warnings.warn(
            "run_screening_pass(y_resid=...) is deprecated; use rhs=... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        rhs = y_resid

    # Stage rhs on the device once. Accept both 1-D (k=1) and 2-D
    # (k>=1) inputs; the screening kernel handles both layouts.
    if rhs is not None:
        rhs_np = np.ascontiguousarray(np.asarray(rhs, dtype=np.float64))
        if rhs_np.ndim not in (1, 2):
            raise ValueError(
                f"rhs must be 1-D or 2-D, got shape {rhs_np.shape}"
            )
        rhs_dev = cp.asarray(rhs_np)
    else:
        rhs_dev = None

    # Validate per-shard sample counts against the intersect (if any).
    if sample_intersect is not None:
        sample_intersect = np.asarray(sample_intersect, dtype=np.int64)
        if sample_intersect.ndim != 1:
            raise ValueError("sample_intersect must be 1D")
        # sample_intersect is passed down to a CUDA kernel as int32 (signed),
        # so any entry >= 2^31 silently truncates to a negative offset and
        # reads garbage. Refuse at the gateway.
        _INT32_MAX = (1 << 31) - 1
        if sample_intersect.size and int(sample_intersect.max()) > _INT32_MAX:
            raise ValueError(
                "sample_intersect contains indices >= 2^31; CUDA kernels read "
                "these as int32 and would silently truncate. Re-shard or "
                "remap so every index fits in int32."
            )
        if sample_intersect.size > _INT32_MAX:
            raise ValueError(
                f"sample_intersect length {sample_intersect.size} exceeds "
                "int32 capacity (2^31-1); CUDA kernel offsets cannot index it."
            )
        for n_s in n_samples_per_path:
            if sample_intersect.size and (
                int(sample_intersect.min()) < 0 or int(sample_intersect.max()) >= n_s
            ):
                raise ValueError(
                    "sample_intersect contains indices outside a shard's sample range"
                )
        n_samples_out = int(sample_intersect.shape[0])
    else:
        # All shards must have the same sample count if no intersect is given.
        n_samples_out = int(n_samples_per_path[0])
        if any(int(n) != n_samples_out for n in n_samples_per_path):
            raise ValueError(
                "sample counts differ across shards; provide sample_intersect"
            )

    if rhs_dev is not None and int(rhs_dev.shape[0]) != n_samples_out:
        raise ValueError(
            f"rhs leading dim {int(rhs_dev.shape[0])} != effective n_samples "
            f"{n_samples_out}"
        )

    per_path_results: list[dict[str, Any]] = []
    for path, n_s, n_v in zip(bed_paths, n_samples_per_path, n_variants_per_path):
        per_path_results.append(
            _screen_one_path(
                Path(path),
                int(n_s),
                int(n_v),
                sample_intersect=sample_intersect,
                rhs_dev=rhs_dev,
                count_a1=count_a1,
                stream=stream,
                chunk_bytes=_DEFAULT_CHUNK_BYTES,
            )
        )

    # Concatenate on-device, then copy to host.
    count = cp.concatenate([r["count"] for r in per_path_results])
    s = cp.concatenate([r["sum"] for r in per_path_results])
    ssq = cp.concatenate([r["sumsq"] for r in per_path_results])
    if rhs_dev is not None:
        drhs = cp.concatenate([r["dosage_rhs"] for r in per_path_results], axis=0)
        orhs = cp.concatenate([r["observed_rhs"] for r in per_path_results], axis=0)
        drhs_host: Any = cp.asnumpy(drhs)
        orhs_host: Any = cp.asnumpy(orhs)
    else:
        drhs_host = None
        orhs_host = None

    return {
        "count": cp.asnumpy(count),
        "sum": cp.asnumpy(s),
        "sumsq": cp.asnumpy(ssq),
        "dosage_rhs": drhs_host,
        "observed_rhs": orhs_host,
    }


def finalize_standardized_rhs(
    dosage_rhs: np.ndarray,
    observed_rhs: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Combine the screening partials into the standardized inner product.

    Given per-variant accumulators ``dosage_rhs`` and ``observed_rhs``
    returned by :func:`run_screening_pass` (or
    :func:`sv_pgs.bitpacked.screening.screen`), plus the per-variant
    ``mean`` and ``scale``, return::

        Z_v . y = (dosage_rhs - mean * observed_rhs) / scale

    Broadcasts over any trailing rhs-column axis (k).

    Parameters
    ----------
    dosage_rhs, observed_rhs
        Shape ``(n_variants,)`` or ``(n_variants, k)``, float64.
    mean, scale
        Shape ``(n_variants,)``, float. ``scale[v] == 0`` columns are
        passed through as 0 (the screening kernel / preprocessing pipeline
        imputes such variants to mean 0 / scale 1 upstream, so this is
        only a defensive safety net).
    """
    mean_arr = np.asarray(mean, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    drhs = np.asarray(dosage_rhs, dtype=np.float64)
    orhs = np.asarray(observed_rhs, dtype=np.float64)
    if drhs.ndim == 2:
        numer = drhs - mean_arr[:, None] * orhs
        safe_scale = np.where(scale_arr == 0.0, 1.0, scale_arr)[:, None]
    else:
        numer = drhs - mean_arr * orhs
        safe_scale = np.where(scale_arr == 0.0, 1.0, scale_arr)
    return numer / safe_scale

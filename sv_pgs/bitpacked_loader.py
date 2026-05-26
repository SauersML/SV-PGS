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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sv_pgs.plink import (
    _BYTE_DECODE_LUT_A1,
    _BYTE_DECODE_LUT_A2,
    _ENCODE_LOOKUP_A1,
    _bytes_per_variant,
    open_bed,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix

# Stream all-variant reads in chunks no larger than this many bytes to bound
# host-side peak RSS when the BED is very large.
_FULL_READ_CHUNK_BYTES: int = 256 * 1024 * 1024


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


def _read_all_packed(
    reader: open_bed,
    n_variants: int,
    bytes_per_variant: int,
    pinned_view: np.ndarray,
) -> None:
    """Stream all variants from the BED into ``pinned_view`` in 256 MB chunks."""
    from sv_pgs.plink import PLINK1_HEADER_SIZE

    total_bytes = n_variants * bytes_per_variant
    if total_bytes == 0:
        return
    chunk_variants = max(1, _FULL_READ_CHUNK_BYTES // max(1, bytes_per_variant))
    cursor = 0
    while cursor < n_variants:
        stop = min(n_variants, cursor + chunk_variants)
        byte_offset = PLINK1_HEADER_SIZE + cursor * bytes_per_variant
        byte_length = (stop - cursor) * bytes_per_variant
        payload = reader._pread_payload(byte_offset, byte_length)
        host_offset = cursor * bytes_per_variant
        pinned_view[host_offset : host_offset + byte_length] = payload
        cursor = stop


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

    for row_start in range(0, n_variants, chunk_rows):
        row_stop = min(n_variants, row_start + chunk_rows)
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
        # Observed dosages are 0/1/2 ints (int8); index the encode LUT.
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
        try:
            raw_total_bytes = gathered_n_variants * src_bpv
            # Read directly into pinned memory when no sample rebitpack is needed
            # to avoid an intermediate copy.
            if sample_indices is None:
                pinned_mem, pinned_view = _allocate_pinned(cp, raw_total_bytes)
                _read_all_packed(reader, gathered_n_variants, src_bpv, pinned_view)
                packed_for_upload = pinned_view
                final_bpv = src_bpv
            else:
                # Read into a plain numpy buffer; rebitpack will produce the
                # final layout sized for the chosen samples.
                raw = np.empty((raw_total_bytes,), dtype=np.uint8)
                _read_all_packed(reader, gathered_n_variants, src_bpv, raw)
                packed_for_upload = raw  # placeholder, replaced below
                final_bpv = src_bpv
        finally:
            # open_bed currently has no explicit close; the fd is process-lived.
            pass
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
        payload = reader._pread_indexed_variant_payload(
            var_idx,
            bytes_per_variant=src_bpv,
        )
        if sample_indices is None:
            # Move the gather result into a pinned staging buffer.
            pinned_mem, pinned_view = _allocate_pinned(cp, payload.nbytes)
            pinned_view[:] = payload
            packed_for_upload = pinned_view
            final_bpv = src_bpv
        else:
            packed_for_upload = payload
            final_bpv = src_bpv  # placeholder, replaced after rebitpack

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

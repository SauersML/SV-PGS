"""Fused MAF + marginal-z screening kernel for bitpacked PLINK genotypes.

One pass over the bitpacked ``(n_variants, bytes_per_variant)`` uint8 buffer
accumulates per-variant:

    count[v]            : int32   -- number of non-missing samples
    sum[v]              : float64 -- sum of raw dosages over non-missing samples
    sumsq[v]            : float64 -- sum of squared raw dosages over non-missing samples
    dosage_rhs[v, j]    : float64 -- sum_{i in obs} d_iv * rhs[i, j]
    observed_rhs[v, j]  : float64 -- sum_{i in obs}        rhs[i, j]

The two right-hand-side accumulators are both required to reconstruct the
standardized inner product under missingness:

    Z_v.T @ rhs[:, j]
        = ( dosage_rhs[v, j] - mean[v] * observed_rhs[v, j] ) / scale[v]

(Use :func:`finalize_standardized_rhs` to perform that combine on-device.)

All outputs are accumulated INTO the caller-provided buffers (the caller
initializes them, typically with ``cupy.zeros``).
"""

from __future__ import annotations

import warnings
from typing import Any

from sv_pgs.bitpacked.launch import gpu_arch, screening_config

# CUDA source for the fused screening kernel. One block per variant. Each
# block iterates over the variant's packed bytes in strides of blockDim.x,
# unpacks four samples per byte via a __constant__ LUT, and accumulates the
# reductions in registers / shared memory. A block-wide reduction at the end
# writes a single value per output per (variant, k) -- no global atomics.
#
# The kernel handles k >= 1 RHS columns by looping inside the kernel and
# performing one block reduction per (column, accumulator). The genotype
# decode work is done ONCE per byte regardless of k.
_KERNEL_SRC = r"""
extern "C" {

// Decode LUT: indexed by [count_a1 (0/1)][byte (0..255)][sample (0..3)].
// Missing slot encoded as -127 to match plink.py PLINK_MISSING_INT8.
__constant__ signed char SCREEN_DECODE_LUT[2][256][4];

#ifndef SCREEN_MAX_K
#define SCREEN_MAX_K 8
#endif

__global__ void bitpacked_screening_kernel(
    const unsigned char* __restrict__ packed,
    const double* __restrict__ rhs,         // (n_samples, k) row-major, may be null
    int* __restrict__ out_count,
    double* __restrict__ out_sum,
    double* __restrict__ out_sumsq,
    double* __restrict__ out_dosage_rhs,    // (n_variants, k), may be null
    double* __restrict__ out_observed_rhs,  // (n_variants, k), may be null
    const int n_samples,
    const int n_variants,
    const int bytes_per_variant,
    const int count_a1,
    const int k,
    const int has_rhs)
{
    const int v = blockIdx.x;
    if (v >= n_variants) return;

    const unsigned char* row = packed + (size_t)v * (size_t)bytes_per_variant;
    const int lut_index = count_a1 ? 1 : 0;

    int    local_count = 0;
    double local_sum   = 0.0;
    double local_sumsq = 0.0;
    // Per-column accumulators kept in registers. Cap is SCREEN_MAX_K; the
    // wrapper validates that the requested k <= SCREEN_MAX_K.
    double local_drhs[SCREEN_MAX_K];
    double local_orhs[SCREEN_MAX_K];
    #pragma unroll
    for (int j = 0; j < SCREEN_MAX_K; ++j) { local_drhs[j] = 0.0; local_orhs[j] = 0.0; }

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // Walk packed bytes for this variant. Each byte covers 4 samples.
    for (int b = tid; b < bytes_per_variant; b += bdim) {
        const unsigned int byte_val = (unsigned int)row[b];
        const int base_sample = b * 4;

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int sample_idx = base_sample + s;
            if (sample_idx >= n_samples) break;  // trailing padding
            const int dose = (int)SCREEN_DECODE_LUT[lut_index][byte_val][s];
            if (dose == -127) continue;  // missing
            const double d = (double)dose;
            local_count += 1;
            local_sum   += d;
            local_sumsq += d * d;
            if (has_rhs) {
                const double* rhs_row = rhs + (size_t)sample_idx * (size_t)k;
                for (int j = 0; j < k; ++j) {
                    const double yj = rhs_row[j];
                    local_drhs[j] += d * yj;
                    local_orhs[j] += yj;
                }
            }
        }
    }

    // Shared memory layout:
    //   [bdim * int32]  count partials
    //   [bdim * f64]    sum partials
    //   [bdim * f64]    sumsq partials
    //   [bdim * f64]    scratch (reused per-k for drhs / orhs)
    extern __shared__ unsigned char smem_raw[];
    int*    s_count   = reinterpret_cast<int*>(smem_raw);
    double* s_sum     = reinterpret_cast<double*>(s_count + bdim);
    double* s_sumsq   = s_sum + bdim;
    double* s_scratch = s_sumsq + bdim;

    s_count[tid] = local_count;
    s_sum[tid]   = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();

    for (int offset = bdim >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_count[tid] += s_count[tid + offset];
            s_sum[tid]   += s_sum[tid + offset];
            s_sumsq[tid] += s_sumsq[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_count[v] += s_count[0];
        out_sum[v]   += s_sum[0];
        out_sumsq[v] += s_sumsq[0];
    }

    if (has_rhs) {
        for (int j = 0; j < k; ++j) {
            // dosage_rhs reduction
            __syncthreads();
            s_scratch[tid] = local_drhs[j];
            __syncthreads();
            for (int offset = bdim >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    s_scratch[tid] += s_scratch[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                out_dosage_rhs[(size_t)v * (size_t)k + (size_t)j] += s_scratch[0];
            }

            // observed_rhs reduction
            __syncthreads();
            s_scratch[tid] = local_orhs[j];
            __syncthreads();
            for (int offset = bdim >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    s_scratch[tid] += s_scratch[tid + offset];
                }
                __syncthreads();
            }
            if (tid == 0) {
                out_observed_rhs[(size_t)v * (size_t)k + (size_t)j] += s_scratch[0];
            }
        }
    }
}

}  // extern "C"
"""


# Must match SCREEN_MAX_K in the CUDA source. If you bump this, you also pay
# the per-thread register cost (2 fp64 slots per extra column) per block.
SCREEN_MAX_K: int = 8

_kernel_cache: dict[str, Any] = {}
_lut_uploaded: dict[int, bool] = {}


def _build_decode_lut_bytes() -> bytes:
    """Build the 2 x 256 x 4 int8 LUT as raw bytes for __constant__ upload.

    Layout: ``lut[count_a1_flag][byte][sample]`` where ``count_a1_flag``
    0 -> A2 convention, 1 -> A1 convention.
    """
    import numpy as np

    # Codes: 0b00, 0b01 (missing), 0b10, 0b11
    # A1 mapping: 00->2, 01->missing, 10->1, 11->0
    # A2 mapping: 00->0, 01->missing, 10->1, 11->2
    a1_map = {0: 2, 1: -127, 2: 1, 3: 0}
    a2_map = {0: 0, 1: -127, 2: 1, 3: 2}
    lut = np.zeros((2, 256, 4), dtype=np.int8)
    for byte_val in range(256):
        codes = [(byte_val >> (2 * k)) & 0b11 for k in range(4)]
        for s, c in enumerate(codes):
            lut[0, byte_val, s] = a2_map[c]
            lut[1, byte_val, s] = a1_map[c]
    return lut.tobytes()


def _get_module() -> Any:
    """Compile (once) and return the CuPy RawModule with LUT uploaded."""
    import cupy as cp  # lazy

    dev_id = int(cp.cuda.runtime.getDevice())
    key = f"screening_module_dev{dev_id}"
    module = _kernel_cache.get(key)
    if module is None:
        module = cp.RawModule(
            code=_KERNEL_SRC,
            options=("--std=c++14", f"-DSCREEN_MAX_K={SCREEN_MAX_K}"),
        )
        _kernel_cache[key] = module
    if not _lut_uploaded.get(dev_id, False):
        lut_bytes = _build_decode_lut_bytes()
        const_ptr = module.get_global("SCREEN_DECODE_LUT")
        const_ptr.copy_from_host(lut_bytes, len(lut_bytes))
        _lut_uploaded[dev_id] = True
    return module


def screen(
    packed: Any,
    n_samples: int,
    out_count: Any,
    out_sum: Any,
    out_sumsq: Any,
    rhs: Any | None = None,
    out_dosage_rhs: Any | None = None,
    out_observed_rhs: Any | None = None,
    count_a1: bool = True,
    stream: Any | None = None,
    *,
    y_resid: Any | None = None,
    out_y_dot: Any | None = None,
) -> None:
    """Fused one-pass screening reduction.

    Parameters
    ----------
    packed : cupy.ndarray, shape (n_variants, bytes_per_variant), dtype uint8
        Bitpacked variant-major buffer (PLINK 1.9 layout, no magic bytes).
    n_samples : int
        Number of valid samples (last 0-3 packed slots may be padding).
    out_count : cupy.ndarray, shape (n_variants,), dtype int32
        Accumulator for non-missing sample counts.
    out_sum : cupy.ndarray, shape (n_variants,), dtype float64
        Accumulator for sum of raw dosages.
    out_sumsq : cupy.ndarray, shape (n_variants,), dtype float64
        Accumulator for sum of squared raw dosages.
    rhs : cupy.ndarray | None, shape (n_samples,) or (n_samples, k), float32/64
        Per-sample right-hand-side vector(s). 1-D inputs are treated as k=1.
        Internally promoted to float64, C-contiguous.
    out_dosage_rhs : cupy.ndarray | None, shape (n_variants, k), dtype float64
        Accumulator for ``sum_{i in obs} d_iv * rhs[i, j]``. Required iff
        ``rhs`` is given. May be passed as ``(n_variants,)`` when k=1.
    out_observed_rhs : cupy.ndarray | None, shape (n_variants, k), dtype float64
        Accumulator for ``sum_{i in obs} rhs[i, j]``. Required iff ``rhs`` is
        given. May be passed as ``(n_variants,)`` when k=1.
    count_a1 : bool
        PLINK1 count-A1 convention (default True, matches sv_pgs).
    stream : cupy.cuda.Stream | None
        Stream to launch on. ``None`` uses the current stream.
    y_resid, out_y_dot : DEPRECATED keyword-only aliases
        Old k=1 form. ``y_resid`` maps to ``rhs`` and ``out_y_dot`` to
        ``out_dosage_rhs``. Callers using this path MUST still supply
        ``out_observed_rhs`` (needed for correct missingness-aware
        standardization). Emits :class:`DeprecationWarning`.
    """
    import cupy as cp  # lazy

    # ---- Deprecated alias handling -----------------------------------------
    if y_resid is not None or out_y_dot is not None:
        warnings.warn(
            "`y_resid`/`out_y_dot` are deprecated; use `rhs`/`out_dosage_rhs` "
            "and supply `out_observed_rhs` to enable correct missingness-aware "
            "standardization.",
            DeprecationWarning,
            stacklevel=2,
        )
        if rhs is not None or out_dosage_rhs is not None:
            raise ValueError(
                "cannot mix deprecated y_resid/out_y_dot with new rhs/out_dosage_rhs"
            )
        if y_resid is None or out_y_dot is None:
            raise ValueError("deprecated path requires BOTH y_resid and out_y_dot")
        if out_observed_rhs is None:
            raise ValueError(
                "deprecated path still requires `out_observed_rhs` to be passed "
                "(needed for correct standardization under missingness)"
            )
        rhs = y_resid
        out_dosage_rhs = out_y_dot

    # ---- Basic shape/dtype validation --------------------------------------
    if packed.dtype != cp.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D, got shape {packed.shape}")
    n_variants, bytes_per_variant = int(packed.shape[0]), int(packed.shape[1])
    expected_bpv = (int(n_samples) + 3) // 4
    if bytes_per_variant < expected_bpv:
        raise ValueError(
            f"bytes_per_variant={bytes_per_variant} too small for n_samples={n_samples}"
        )

    if out_count.dtype != cp.int32:
        raise TypeError(f"out_count must be int32, got {out_count.dtype}")
    if out_sum.dtype != cp.float64:
        raise TypeError(f"out_sum must be float64, got {out_sum.dtype}")
    if out_sumsq.dtype != cp.float64:
        raise TypeError(f"out_sumsq must be float64, got {out_sumsq.dtype}")
    for name, arr in (("out_count", out_count), ("out_sum", out_sum), ("out_sumsq", out_sumsq)):
        if arr.shape != (n_variants,):
            raise ValueError(f"{name} must have shape ({n_variants},), got {arr.shape}")

    # ---- RHS handling ------------------------------------------------------
    has_rhs = rhs is not None
    k = 0
    rhs_dev: Any = None  # float64, (n_samples, k), C-contiguous
    if has_rhs:
        if out_dosage_rhs is None or out_observed_rhs is None:
            raise ValueError(
                "out_dosage_rhs and out_observed_rhs are both required when rhs is provided"
            )

        if rhs.ndim == 1:
            if rhs.shape != (n_samples,):
                raise ValueError(
                    f"1-D rhs must have shape ({n_samples},), got {rhs.shape}"
                )
            k = 1
            rhs_2d = rhs.reshape(n_samples, 1)
        elif rhs.ndim == 2:
            if rhs.shape[0] != n_samples:
                raise ValueError(
                    f"rhs.shape[0] must equal n_samples={n_samples}, got {rhs.shape}"
                )
            k = int(rhs.shape[1])
            rhs_2d = rhs
        else:
            raise ValueError(f"rhs must be 1-D or 2-D, got {rhs.ndim}-D")

        if k < 1:
            raise ValueError("rhs must have at least one column")
        if k > SCREEN_MAX_K:
            raise ValueError(
                f"k={k} exceeds SCREEN_MAX_K={SCREEN_MAX_K}; "
                "rebuild module with larger SCREEN_MAX_K"
            )

        if rhs_2d.dtype != cp.float64:
            rhs_dev = cp.ascontiguousarray(rhs_2d, dtype=cp.float64)
        else:
            rhs_dev = cp.ascontiguousarray(rhs_2d)

        # Normalize accumulator shapes: accept (n_variants,) when k==1.
        def _check_out(name: str, arr: Any) -> Any:
            if arr.dtype != cp.float64:
                raise TypeError(f"{name} must be float64, got {arr.dtype}")
            if k == 1 and arr.shape == (n_variants,):
                return arr.reshape(n_variants, 1)
            if arr.shape != (n_variants, k):
                raise ValueError(
                    f"{name} must have shape ({n_variants}, {k}), got {arr.shape}"
                )
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            return arr

        out_dosage_rhs_v = _check_out("out_dosage_rhs", out_dosage_rhs)
        out_observed_rhs_v = _check_out("out_observed_rhs", out_observed_rhs)

        rhs_ptr = int(rhs_dev.data.ptr)
        out_drhs_ptr = int(out_dosage_rhs_v.data.ptr)
        out_orhs_ptr = int(out_observed_rhs_v.data.ptr)
    else:
        if out_dosage_rhs is not None or out_observed_rhs is not None:
            raise ValueError(
                "out_dosage_rhs / out_observed_rhs must be None when rhs is None"
            )
        rhs_ptr = 0
        out_drhs_ptr = 0
        out_orhs_ptr = 0

    # ---- Launch config -----------------------------------------------------
    cfg = screening_config(int(n_samples), n_variants, gpu_arch())
    grid = cfg["grid"]
    block = cfg["block"]
    block_x = int(block[0])

    # Required shmem: bdim * (int32 + 3 * float64) = bdim * 28 bytes.
    shmem_needed = block_x * (4 + 3 * 8)

    module = _get_module()
    kernel = module.get_function("bitpacked_screening_kernel")

    args = (
        packed,
        cp.uint64(rhs_ptr),
        out_count,
        out_sum,
        out_sumsq,
        cp.uint64(out_drhs_ptr),
        cp.uint64(out_orhs_ptr),
        cp.int32(int(n_samples)),
        cp.int32(n_variants),
        cp.int32(bytes_per_variant),
        cp.int32(1 if count_a1 else 0),
        cp.int32(k),
        cp.int32(1 if has_rhs else 0),
    )

    if stream is None:
        kernel(grid, block, args, shared_mem=shmem_needed)
    else:
        with stream:
            kernel(grid, block, args, shared_mem=shmem_needed)


def finalize_standardized_rhs(
    dosage_rhs: Any,
    observed_rhs: Any,
    mean: Any,
    scale: Any,
    out: Any | None = None,
) -> Any:
    """Compute ``(dosage_rhs - mean[:, None] * observed_rhs) / scale[:, None]``.

    Parameters
    ----------
    dosage_rhs : cupy.ndarray, shape (n_variants,) or (n_variants, k)
    observed_rhs : cupy.ndarray, same shape as ``dosage_rhs``
    mean : cupy.ndarray, shape (n_variants,)
    scale : cupy.ndarray, shape (n_variants,)
        Per-variant standard deviation. Zero entries should already be
        imputed to 1.0 by the caller (matches sv_pgs convention).
    out : cupy.ndarray | None
        Optional output buffer with the same shape and dtype as ``dosage_rhs``.

    Returns
    -------
    cupy.ndarray
        ``Z_v.T @ rhs`` per variant (and per RHS column when 2-D).
    """
    import cupy as cp  # lazy

    if dosage_rhs.shape != observed_rhs.shape:
        raise ValueError(
            f"dosage_rhs and observed_rhs must share shape; got "
            f"{dosage_rhs.shape} vs {observed_rhs.shape}"
        )

    if dosage_rhs.ndim == 1:
        m = mean
        s = scale
    elif dosage_rhs.ndim == 2:
        m = mean[:, None]
        s = scale[:, None]
    else:
        raise ValueError(f"dosage_rhs must be 1-D or 2-D, got {dosage_rhs.ndim}-D")

    if mean.shape[0] != dosage_rhs.shape[0]:
        raise ValueError(
            f"mean.shape[0]={mean.shape[0]} must equal n_variants={dosage_rhs.shape[0]}"
        )
    if scale.shape[0] != dosage_rhs.shape[0]:
        raise ValueError(
            f"scale.shape[0]={scale.shape[0]} must equal n_variants={dosage_rhs.shape[0]}"
        )

    if out is None:
        return ((dosage_rhs - m * observed_rhs) / s).astype(dosage_rhs.dtype, copy=False)

    if out.shape != dosage_rhs.shape:
        raise ValueError(
            f"out must have shape {dosage_rhs.shape}, got {out.shape}"
        )
    cp.subtract(dosage_rhs, m * observed_rhs, out=out)
    cp.divide(out, s, out=out)
    return out

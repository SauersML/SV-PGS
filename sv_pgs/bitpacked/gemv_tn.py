"""Transpose bitpacked GEMV: g[v] += sum_i Z[i,v] * y[i].

Design: variant-tile per block, warp per variant, per-variant 4-entry
coefficient table cached in shared memory.

Key insight: every standardized contribution is one of 4 values per variant
(one per 2-bit code). Precomputing
``coef[code] = (raw_dosage(code) - mean) / std`` (with the missing slot set
to 0) means the inner loop becomes a single shared-memory lookup + FMA per
sample, with NO branches and NO constant-memory thrash. This avoids:

- Repeated constant-memory LUT lookups serialized across warp lanes (the
  baseline's bottleneck on V100).
- Branch-heavy missing-value handling inside the unrolled inner loop.
- High register pressure from broad per-thread inner unrolls.

Layout:
- Each block owns ``VPB`` contiguous variants (default 8), one variant per
  warp. Block = 256 threads.
- The block's shared memory holds: (a) a CHUNK_S-float y tile, and (b) a
  4-float coefficient table per warp (4 * VPB = 32 floats total).
- Each warp's 32 lanes stride along the variant's packed-byte row at
  warp-width stride; per outer iter each lane reads ``BPL = 4`` bytes,
  giving 16 sample contributions per lane per iter and 512 samples per
  warp per iter.
- Per-thread accumulator -> warp ``__shfl_xor`` reduction -> lane 0 writes
  ``out[v] += partial`` (no atomics, each variant is owned by one warp).
"""

from __future__ import annotations

import functools
from typing import Any

from sv_pgs.bitpacked.launch import gemv_tn_config, gpu_arch

_KERNEL_NAME = "bitpacked_gemv_tn_kernel"

_PLINK_MISSING_INT8 = -127

# Tile parameters baked into the kernel via -D macros.
_VPB = 8                 # variants per block, one warp each
_BPL = 4                 # bytes per lane per outer iter (16 samples)
_WARP = 32
_BLOCK = _VPB * _WARP    # 256 threads
_CHUNK_B = _BPL * _WARP  # 128 bytes per outer iter (per row)
_CHUNK_S = _CHUNK_B * 4  # 512 samples per outer iter

_CUDA_SOURCE = r"""
extern "C" {

#ifndef VPB
#define VPB 8
#endif
#ifndef BPL
#define BPL 4
#endif

#define WARP 32
#define BLOCK (VPB * WARP)
#define CHUNK_B (BPL * WARP)
#define CHUNK_S (CHUNK_B * 4)

// Raw dosages per 2-bit code, per count_a1 setting (missing -> 0 placeholder;
// we zero the coefficient explicitly so this constant is unused for code=1).
//   count_a1=true : 00->2, 01->miss, 10->1, 11->0
//   count_a1=false: 00->0, 01->miss, 10->1, 11->2

__global__ void bitpacked_gemv_tn_kernel(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ std_,
    float* __restrict__ out,
    const int n_samples,
    const int n_variants,
    const int bytes_per_variant,
    const int count_a1)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int v_global = blockIdx.x * VPB + warp_id;
    const int v_valid = (v_global < n_variants);

    extern __shared__ float smem[];
    // Layout: [CHUNK_S floats: y tile][VPB * 4 floats: per-warp coef table]
    float* y_tile = smem;
    float* coef_tbl = smem + CHUNK_S;  // size VPB * 4

    // Per-warp coefficient table: lane 0..3 of each warp loads its variant's
    // (raw_dosage(code) - mean) / std for code in 0..3, with code=1 (missing)
    // explicitly zeroed.
    if (lane < 4) {
        float c = 0.0f;
        if (v_valid) {
            const float mv = mean[v_global];
            const float sv = std_[v_global];
            const float inv_s = (sv > 0.0f) ? __frcp_rn(sv) : 0.0f;
            float raw = 0.0f;
            // code -> raw dosage table
            if (count_a1) {
                if      (lane == 0) raw = 2.0f;
                else if (lane == 2) raw = 1.0f;
                else if (lane == 3) raw = 0.0f;
                // lane == 1 -> missing -> handled below
            } else {
                if      (lane == 0) raw = 0.0f;
                else if (lane == 2) raw = 1.0f;
                else if (lane == 3) raw = 2.0f;
            }
            if (lane == 1) {
                c = 0.0f;  // missing -> zero contribution
            } else {
                c = (raw - mv) * inv_s;
            }
        }
        coef_tbl[warp_id * 4 + lane] = c;
    }
    __syncthreads();

    // Each warp loads its 4 coefficients into registers (broadcast through
    // shmem within a warp is free for repeated identical addresses, but we
    // pull them into registers so the inner loop is shmem-bank-conflict-free).
    const float* my_coef = coef_tbl + warp_id * 4;
    const float c0 = my_coef[0];
    const float c1 = my_coef[1];
    const float c2 = my_coef[2];
    const float c3 = my_coef[3];

    const size_t row_off = (size_t)v_global * (size_t)bytes_per_variant;
    const unsigned char* row = packed + row_off;

    float acc = 0.0f;

    for (int s0 = 0; s0 < n_samples; s0 += CHUNK_S) {
        // Cooperative y-tile load with zero-padding past n_samples.
        #pragma unroll
        for (int k = tid; k < CHUNK_S; k += BLOCK) {
            const int s = s0 + k;
            y_tile[k] = (s < n_samples) ? y[s] : 0.0f;
        }
        __syncthreads();

        if (v_valid) {
            const int byte_base = s0 >> 2;
            const int valid_bytes = bytes_per_variant - byte_base;
            const int chunk_b = (valid_bytes < CHUNK_B) ? valid_bytes : CHUNK_B;

            if (chunk_b == CHUNK_B) {
                // Hot path: full chunk, no per-iter branches.
                #pragma unroll
                for (int j = 0; j < BPL; ++j) {
                    const int local_b = lane + j * WARP;
                    const int global_b = byte_base + local_b;
                    const unsigned int bv = (unsigned int)__ldg(row + global_b);
                    const int s_local_byte = local_b << 2;
                    // 4 samples per byte. Coefficient via in-register select.
                    #pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const unsigned int code = (bv >> (2 * slot)) & 0x3u;
                        const float coef = (code == 0u) ? c0
                                         : (code == 1u) ? c1
                                         : (code == 2u) ? c2 : c3;
                        acc += coef * y_tile[s_local_byte + slot];
                    }
                }
            } else {
                // Tail chunk: bound-check byte index; y_tile is already
                // zero-padded past n_samples.
                #pragma unroll
                for (int j = 0; j < BPL; ++j) {
                    const int local_b = lane + j * WARP;
                    if (local_b >= chunk_b) break;
                    const int global_b = byte_base + local_b;
                    const unsigned int bv = (unsigned int)__ldg(row + global_b);
                    const int s_local_byte = local_b << 2;
                    #pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const unsigned int code = (bv >> (2 * slot)) & 0x3u;
                        const float coef = (code == 0u) ? c0
                                         : (code == 1u) ? c1
                                         : (code == 2u) ? c2 : c3;
                        acc += coef * y_tile[s_local_byte + slot];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Warp reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, offset);
    }

    if (lane == 0 && v_valid) {
        out[v_global] += acc;
    }
}

}  // extern "C"
"""


@functools.lru_cache(maxsize=4)
def _get_kernel(_key: tuple[Any, ...]) -> Any:
    import cupy as cp

    options = (
        "--std=c++14",
        f"-DVPB={_VPB}",
        f"-DBPL={_BPL}",
    )
    module = cp.RawModule(code=_CUDA_SOURCE, options=options)
    return module.get_function("bitpacked_gemv_tn_kernel")


def gemv_tn(
    packed: Any,
    n_samples: int,
    y: Any,
    mean: Any,
    std: Any,
    out: Any,
    count_a1: bool = True,
    stream: Any = None,
) -> None:
    """Transpose bitpacked GEMV.

    Computes ``g[v] += sum_i ((dosage[i,v] - mean[v]) / std[v]) * y[i]``
    where missing genotypes contribute 0. ``out`` is accumulated INTO.
    """
    import cupy as cp  # lazy

    if packed.dtype != cp.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if y.dtype != cp.float32:
        raise TypeError(f"y must be float32, got {y.dtype}")
    if mean.dtype != cp.float32:
        raise TypeError(f"mean must be float32, got {mean.dtype}")
    if std.dtype != cp.float32:
        raise TypeError(f"std must be float32, got {std.dtype}")
    if out.dtype != cp.float32:
        raise TypeError(f"out must be float32, got {out.dtype}")

    if packed.ndim != 2:
        raise ValueError(
            f"packed must be 2-D (n_variants, bytes_per_variant), got shape {packed.shape}"
        )

    n_variants, bytes_per_variant = int(packed.shape[0]), int(packed.shape[1])
    expected_bpv = (int(n_samples) + 3) // 4
    if bytes_per_variant != expected_bpv:
        raise ValueError(
            f"bytes_per_variant {bytes_per_variant} != (n_samples + 3) // 4 = {expected_bpv}"
        )

    if y.shape != (n_samples,):
        raise ValueError(f"y shape {y.shape} != ({n_samples},)")
    if mean.shape != (n_variants,):
        raise ValueError(f"mean shape {mean.shape} != ({n_variants},)")
    if std.shape != (n_variants,):
        raise ValueError(f"std shape {std.shape} != ({n_variants},)")
    if out.shape != (n_variants,):
        raise ValueError(f"out shape {out.shape} != ({n_variants},)")

    if not packed.flags.c_contiguous:
        raise ValueError("packed must be C-contiguous")

    if n_variants == 0 or n_samples == 0:
        return

    _ = gemv_tn_config(int(n_samples), int(n_variants), gpu_arch())

    grid_x = (n_variants + _VPB - 1) // _VPB
    grid = (grid_x, 1, 1)
    block = (_BLOCK, 1, 1)
    shmem_bytes = (_CHUNK_S + _VPB * 4) * 4  # y tile + per-warp coef table

    kernel = _get_kernel(("v4", _VPB, _BPL))

    args = (
        packed,
        y,
        mean,
        std,
        out,
        cp.int32(n_samples),
        cp.int32(n_variants),
        cp.int32(bytes_per_variant),
        cp.int32(1 if count_a1 else 0),
    )

    if stream is None:
        kernel(grid, block, args, shared_mem=shmem_bytes)
    else:
        with stream:
            kernel(grid, block, args, shared_mem=shmem_bytes)

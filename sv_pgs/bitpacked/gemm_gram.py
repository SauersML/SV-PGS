"""LD-block gram kernel: B[u, v] += sum_i Z[i, u] * Z[i, v].

Z is the standardized variant matrix:
    Z[i, v] = (dosage[i, v] - mean[v]) / std[v]   if not missing
    Z[i, v] = 0                                    if missing

This module exposes a per-architecture-dispatched ``gemm_gram`` wrapper. Three
named RawKernels are exposed (per spec); for the first cut all three use a
correctness-prioritized fp32 GEMM path with vectorized shared-memory tiles.
The DP4A / TF32 mma / FP16 mma fast paths can replace the kernel bodies later
without changing the wrapper contract.

Per-arch outer tile sizing decisions (matched in ``launch.gemm_gram_config``):
  * Volta      (SM 7.0):  128 x 128 variants, k-tile = 32 samples, ~28 KB static
                           SMEM (0 dynamic). 8 warps, 2x4 WMMA fp16->fp32 frags/warp
                           (m16n16k16). Targets V100 tensor cores (125 TFLOPS peak).
  * T4         (SM 7.5):  64 x 64 variants, k-tile = 32 samples, ~16 KB shmem.
  * Ampere     (SM 8.x):  128 x 128 variants, k-tile = 64 samples, ~64 KB shmem.
  * Hopper     (SM 9.x):  128 x 256 variants, k-tile = 64 samples, ~96 KB shmem.

Each block computes one (tile_m x tile_n) output tile via:
    1. cooperatively load packed bytes for the tile's row variants into shmem,
    2. unpack to fp32 standardized values in registers,
    3. accumulate into a register tile,
    4. atomicAdd-style write-add into ``out`` (out is accumulated INTO).

Symmetric output: only blocks with (block_row_tile <= block_col_tile) do work;
blocks above the diagonal mirror their results into out[v, u] as well.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from sv_pgs.bitpacked import launch as _launch

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    import cupy as cp  # noqa: F401


# ---------------------------------------------------------------------------
# CUDA source — three named kernels, all fp32-fallback for first cut.
# ---------------------------------------------------------------------------
#
# Tile parameters are baked in per-kernel via template macros so that all three
# kernel names exist as independent compiled entry points.
#
# Each kernel signature:
#   __global__ void name(
#       const uint8_t* __restrict__ packed,
#       int n_variants,
#       int n_samples,
#       int bytes_per_variant,
#       const float* __restrict__ mean,
#       const float* __restrict__ std,
#       float* __restrict__ out,
#       int count_a1)
#

_GEMM_GRAM_SRC = r"""
// ---------------------------------------------------------------------------
// Shared device helpers.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int decode_dosage(int code, int count_a1) {
    // Returns -1 to signal missing, else dosage in {0, 1, 2}.
    if (code == 0x1) return -1;
    if (count_a1) {
        // 00->2, 10->1, 11->0
        if (code == 0x0) return 2;
        if (code == 0x2) return 1;
        return 0;  // 0x3
    } else {
        // 00->0, 10->1, 11->2
        if (code == 0x0) return 0;
        if (code == 0x2) return 1;
        return 2;
    }
}

// Compute Z[i, v] for sample i, variant v given packed row pointer and
// per-variant mean/std. Sample i may be >= n_samples (padding) -> returns 0.
__device__ __forceinline__ float z_of(
    const unsigned char* __restrict__ row,
    int i,
    int n_samples,
    float m,
    float s)
{
    if (i >= n_samples) return 0.0f;
    int byte_idx = i >> 2;
    int slot = i & 0x3;
    int code = (row[byte_idx] >> (slot * 2)) & 0x3;
    int d = decode_dosage(code, 1);  // overridden below via wrapper macro
    if (d < 0) return 0.0f;
    return ((float)d - m) / s;
}

// Variant of z_of that respects count_a1 flag.
__device__ __forceinline__ float z_of_flag(
    const unsigned char* __restrict__ row,
    int i,
    int n_samples,
    float m,
    float s,
    int count_a1)
{
    if (i >= n_samples) return 0.0f;
    int byte_idx = i >> 2;
    int slot = i & 0x3;
    int code = (row[byte_idx] >> (slot * 2)) & 0x3;
    int d = decode_dosage(code, count_a1);
    if (d < 0) return 0.0f;
    // Zero-scale columns contribute 0 to the gram (mean-imputed constant column).
    if (!(s > 0.0f)) return 0.0f;
    return ((float)d - m) / s;
}

// ---------------------------------------------------------------------------
// Generic fp32 gram tile kernel template (implemented as a macro because
// CuPy RawKernels don't support C++ templates across modules robustly).
// ---------------------------------------------------------------------------

#define DEFINE_GRAM_KERNEL(NAME, TILE_M, TILE_N, TILE_K, NTHREADS, MTH_C)      \
__global__ void NAME(                                                          \
    const unsigned char* __restrict__ packed,                                  \
    int n_variants,                                                            \
    int n_samples,                                                             \
    int bytes_per_variant,                                                     \
    const float* __restrict__ mean,                                            \
    const float* __restrict__ std,                                             \
    float* __restrict__ out,                                                   \
    int count_a1)                                                              \
{                                                                              \
    const int bx = blockIdx.x;  /* tile row in variants (TILE_M) */            \
    const int by = blockIdx.y;  /* tile col in variants (TILE_N) */            \
    /* Symmetric: only compute u<=v tiles; tile (bx,by) covers rows           \
       [bx*TILE_M, (bx+1)*TILE_M) and cols [by*TILE_N, (by+1)*TILE_N).        \
       We skip when the entire row range is strictly above the col range.  */ \
    const int row0 = bx * TILE_M;                                              \
    const int col0 = by * TILE_N;                                              \
    if (row0 >= n_variants || col0 >= n_variants) return;                      \
    /* Skip strictly-above-diagonal tiles (row0 > col0 + TILE_N - 1). */       \
    if (row0 > col0 + TILE_N - 1) return;                                      \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int nthreads = NTHREADS;                                             \
                                                                               \
    /* Shared mem layout: A[TILE_M][TILE_K] + B[TILE_N][TILE_K] fp32. */       \
    __shared__ float sA[TILE_M][TILE_K];                                       \
    __shared__ float sB[TILE_N][TILE_K];                                       \
                                                                               \
    /* Per-thread output tile: thread (tx, ty) covers output positions       \
       (r, c) with tx = tid % MTH_C, ty = tid / MTH_C.                        \
       NTH = NTHREADS / MTH_C. All compile-time. */                            \
    const int MTH = MTH_C;                                                     \
    const int NTH = NTHREADS / MTH_C;                                          \
    const int tx = tid % MTH;                                                  \
    const int ty = tid / MTH;                                                  \
                                                                               \
    /* Compile-time row/col counts per thread. */                              \
    constexpr int RPT_C = (TILE_M + MTH_C - 1) / MTH_C;                        \
    constexpr int NTH_C = NTHREADS / MTH_C;                                    \
    constexpr int CPT_C = (TILE_N + NTH_C - 1) / NTH_C;                        \
    const int RPT = RPT_C;                                                     \
    const int CPT = CPT_C;                                                     \
                                                                               \
    float acc[RPT_C * CPT_C];                                                  \
    for (int q = 0; q < RPT_C * CPT_C; ++q) acc[q] = 0.0f;                     \
                                                                               \
    /* Loop over k-tiles. */                                                   \
    for (int k0 = 0; k0 < n_samples; k0 += TILE_K) {                           \
        /* Load A tile: TILE_M variants x TILE_K samples (standardized). */    \
        for (int idx = tid; idx < TILE_M * TILE_K; idx += nthreads) {          \
            int rr = idx / TILE_K;                                             \
            int kk = idx % TILE_K;                                             \
            int v = row0 + rr;                                                 \
            int s_idx = k0 + kk;                                               \
            float val = 0.0f;                                                  \
            if (v < n_variants && s_idx < n_samples) {                         \
                const unsigned char* row = packed                              \
                    + (size_t)v * (size_t)bytes_per_variant;                   \
                val = z_of_flag(row, s_idx, n_samples,                         \
                                mean[v], std[v], count_a1);                    \
            }                                                                  \
            sA[rr][kk] = val;                                                  \
        }                                                                      \
        /* Load B tile: TILE_N variants x TILE_K samples (standardized). */    \
        for (int idx = tid; idx < TILE_N * TILE_K; idx += nthreads) {          \
            int cc = idx / TILE_K;                                             \
            int kk = idx % TILE_K;                                             \
            int v = col0 + cc;                                                 \
            int s_idx = k0 + kk;                                               \
            float val = 0.0f;                                                  \
            if (v < n_variants && s_idx < n_samples) {                         \
                const unsigned char* row = packed                              \
                    + (size_t)v * (size_t)bytes_per_variant;                   \
                val = z_of_flag(row, s_idx, n_samples,                         \
                                mean[v], std[v], count_a1);                    \
            }                                                                  \
            sB[cc][kk] = val;                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        /* Accumulate this k-tile. */                                          \
        int qi = 0;                                                            \
        for (int ri = 0; ri < RPT; ++ri) {                                     \
            int rr = tx + ri * MTH;                                            \
            for (int ci = 0; ci < CPT; ++ci) {                                 \
                int cc = ty + ci * NTH;                                        \
                float s = 0.0f;                                                \
                if (rr < TILE_M && cc < TILE_N) {                              \
                    _Pragma("unroll 8")                                        \
                    for (int kk = 0; kk < TILE_K; ++kk) {                      \
                        s += sA[rr][kk] * sB[cc][kk];                          \
                    }                                                          \
                }                                                              \
                acc[qi++] += s;                                                \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Write tile back into out (accumulated INTO). Mirror across diag. */     \
    int qi = 0;                                                                \
    for (int ri = 0; ri < RPT; ++ri) {                                         \
        int rr = tx + ri * MTH;                                                \
        for (int ci = 0; ci < CPT; ++ci) {                                     \
            int cc = ty + ci * NTH;                                            \
            float v = acc[qi++];                                               \
            if (rr < TILE_M && cc < TILE_N) {                                  \
                int u = row0 + rr;                                             \
                int w = col0 + cc;                                             \
                if (u < n_variants && w < n_variants) {                        \
                    if (u <= w) {                                              \
                        atomicAdd(&out[(size_t)u * (size_t)n_variants + w],    \
                                  v);                                          \
                        if (u != w) {                                          \
                            atomicAdd(                                         \
                                &out[(size_t)w * (size_t)n_variants + u], v);  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
"""

# Per-kernel instantiation tails. Each is appended to ``_GEMM_GRAM_PRELUDE``
# to form a complete translation unit that defines exactly ONE entry point.
# This keeps Volta's compile from choking on the high-shmem Ampere/Hopper
# kernels it would never launch.
_KERNEL_TAILS: dict[str, str] = {
    # T4 / Turing (SM 7.5): tile 64x64, k-tile 32, 128 threads, MTH=16 -> NTH=8.
    "bitpacked_gemm_gram_dp4a_kernel": (
        'extern "C" {\n'
        "DEFINE_GRAM_KERNEL(bitpacked_gemm_gram_dp4a_kernel,  64,  64, 32, 128, 16)\n"
        "}\n"
    ),
    # fp32 fallback (any arch): identical params to the dp4a tile.
    "bitpacked_gemm_gram_fp32_kernel": (
        'extern "C" {\n'
        "DEFINE_GRAM_KERNEL(bitpacked_gemm_gram_fp32_kernel,  64,  64, 32, 128, 16)\n"
        "}\n"
    ),
    # Ampere (SM 8.x): tile 128x128, k-tile 64, 256 threads, MTH=16 -> NTH=16.
    "bitpacked_gemm_gram_tf32_kernel": (
        'extern "C" {\n'
        "DEFINE_GRAM_KERNEL(bitpacked_gemm_gram_tf32_kernel, 128, 128, 64, 256, 16)\n"
        "}\n"
    ),
    # Hopper (SM 9.x): tile 128x256, k-tile 64, 256 threads, MTH=16 -> NTH=16.
    "bitpacked_gemm_gram_fp16_kernel": (
        'extern "C" {\n'
        "DEFINE_GRAM_KERNEL(bitpacked_gemm_gram_fp16_kernel, 128, 256, 64, 256, 16)\n"
        "}\n"
    ),
}

# Volta WMMA kernel is a standalone TU (does not use the DEFINE_GRAM_KERNEL
# macro). It needs <mma.h> and a different shmem layout.
_VOLTA_KERNEL_NAME = "bitpacked_gemm_gram_volta_mma_kernel"
_VOLTA_V2_KERNEL_NAME = "bitpacked_gemm_gram_volta_mma_v2_kernel"

_GEMM_GRAM_PRELUDE = _GEMM_GRAM_SRC


# ---------------------------------------------------------------------------
# Volta WMMA v2 kernel (sm_70). Identical math to v1, but with software-
# pipelined double-buffered SMEM loads: the next k-tile's bytes are issued
# to SMEM BEFORE the current k-tile's FMAs, so the LSU + tensor-core
# pipelines overlap. Volta has no cp.async, so the prefetch is just a
# manually-scheduled set of ordinary global->shmem loads that the SM's
# warp scheduler interleaves with the WMMA mma_sync calls.
#
# Total SMEM (per block):
#   sA[2 * VOLTA_TILE_M * VOLTA_SMEM_K_STRIDE]  = 2 * 128 * 40 * 2 B = 20480 B
#   sB[2 * VOLTA_TILE_N * VOLTA_SMEM_K_STRIDE]  = 2 * 128 * 40 * 2 B = 20480 B
#   frag_scratch[VOLTA_NWARPS][16*16] float     = 8 * 256 * 4 B     =  8192 B
#   Total ~48 KB — at Volta's static per-block limit. The wrapper sets the
#   cudaFuncAttributeMaxDynamicSharedMemorySize attribute (Volta supports
#   the 96 KB opt-in carveout) so the kernel can be launched safely.
# ---------------------------------------------------------------------------

_GEMM_GRAM_VOLTA_V2_SRC = r"""
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__device__ __forceinline__ int volta2_decode_dosage(int code, int count_a1) {
    if (code == 0x1) return -1;
    if (count_a1) {
        if (code == 0x0) return 2;
        if (code == 0x2) return 1;
        return 0;
    } else {
        if (code == 0x0) return 0;
        if (code == 0x2) return 1;
        return 2;
    }
}

__device__ __forceinline__ float volta2_z_of(
    const unsigned char* __restrict__ row,
    int i,
    int n_samples,
    float m,
    float s,
    int count_a1)
{
    if (i >= n_samples) return 0.0f;
    int byte_idx = i >> 2;
    int slot = i & 0x3;
    int code = (row[byte_idx] >> (slot * 2)) & 0x3;
    int d = volta2_decode_dosage(code, count_a1);
    if (d < 0) return 0.0f;
    if (!(s > 0.0f)) return 0.0f;
    return ((float)d - m) / s;
}

#define V2_TILE_M 128
#define V2_TILE_N 128
#define V2_TILE_K 32
#define V2_WMMA_M 16
#define V2_WMMA_N 16
#define V2_WMMA_K 16
#define V2_NWARPS 8
#define V2_NTHREADS (V2_NWARPS * 32)
#define V2_WY 4
#define V2_WX 2
#define V2_ROWS_PER_WARP (V2_TILE_M / V2_WY)        /* 32 */
#define V2_COLS_PER_WARP (V2_TILE_N / V2_WX)        /* 64 */
#define V2_FRAGS_M (V2_ROWS_PER_WARP / V2_WMMA_M)   /*  2 */
#define V2_FRAGS_N (V2_COLS_PER_WARP / V2_WMMA_N)   /*  4 */
#define V2_KSUB (V2_TILE_K / V2_WMMA_K)             /*  2 */
#define V2_SMEM_K_STRIDE (V2_TILE_K + 8)            /* 40 */
#define V2_TILE_M_STRIDE (V2_TILE_M * V2_SMEM_K_STRIDE)

extern "C" __global__ void bitpacked_gemm_gram_volta_mma_v2_kernel(
    const unsigned char* __restrict__ packed,
    int n_variants,
    int n_samples,
    int bytes_per_variant,
    const float* __restrict__ mean,
    const float* __restrict__ std_,
    float* __restrict__ out,
    int count_a1)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int row0 = bx * V2_TILE_M;
    const int col0 = by * V2_TILE_N;
    if (row0 >= n_variants || col0 >= n_variants) return;
    if (row0 > col0 + V2_TILE_N - 1) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int wy = warp_id / V2_WX;
    const int wx = warp_id % V2_WX;
    const int lane_id = tid & 31;

    // Dynamic shared memory laid out as:
    //   [0 .. 2*M_STRIDE)        : sA[2][...]
    //   [2*M_STRIDE .. 4*M_STRIDE): sB[2][...]
    extern __shared__ __half smem[];
    __half* sA_buf[2] = {smem + 0, smem + V2_TILE_M_STRIDE};
    __half* sB_buf[2] = {
        smem + 2 * V2_TILE_M_STRIDE,
        smem + 3 * V2_TILE_M_STRIDE,
    };

    // Per-warp scratch lives in static SMEM (only 8 KB total).
    __shared__ float frag_scratch[V2_NWARPS][V2_WMMA_M * V2_WMMA_N];

    wmma::fragment<wmma::accumulator, V2_WMMA_M, V2_WMMA_N, V2_WMMA_K, float>
        acc[V2_FRAGS_M][V2_FRAGS_N];
    #pragma unroll
    for (int i = 0; i < V2_FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < V2_FRAGS_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int n_ktiles = (n_samples + V2_TILE_K - 1) / V2_TILE_K;
    if (n_ktiles <= 0) return;

    // ---- Prologue: load k-tile 0 into buffer 0 -------------------------------
    {
        int k0 = 0;
        #pragma unroll
        for (int idx = tid; idx < V2_TILE_M * V2_TILE_K; idx += V2_NTHREADS) {
            int rr = idx / V2_TILE_K;
            int kk = idx % V2_TILE_K;
            int v = row0 + rr;
            int s_idx = k0 + kk;
            float val = 0.0f;
            if (v < n_variants && s_idx < n_samples) {
                const unsigned char* row = packed + (size_t)v * (size_t)bytes_per_variant;
                val = volta2_z_of(row, s_idx, n_samples, mean[v], std_[v], count_a1);
            }
            sA_buf[0][rr * V2_SMEM_K_STRIDE + kk] = __float2half(val);
        }
        #pragma unroll
        for (int idx = tid; idx < V2_TILE_N * V2_TILE_K; idx += V2_NTHREADS) {
            int cc = idx / V2_TILE_K;
            int kk = idx % V2_TILE_K;
            int v = col0 + cc;
            int s_idx = k0 + kk;
            float val = 0.0f;
            if (v < n_variants && s_idx < n_samples) {
                const unsigned char* row = packed + (size_t)v * (size_t)bytes_per_variant;
                val = volta2_z_of(row, s_idx, n_samples, mean[v], std_[v], count_a1);
            }
            sB_buf[0][cc * V2_SMEM_K_STRIDE + kk] = __float2half(val);
        }
    }
    __syncthreads();

    // ---- Steady-state loop --------------------------------------------------
    for (int kidx = 0; kidx < n_ktiles; ++kidx) {
        const int cur = kidx & 1;
        const int nxt = (kidx + 1) & 1;
        const bool have_next = (kidx + 1) < n_ktiles;

        // Issue the next-tile loads BEFORE the FMAs of the current tile, so
        // the LSU pipeline overlaps with WMMA execution.
        if (have_next) {
            int k0_n = (kidx + 1) * V2_TILE_K;
            #pragma unroll
            for (int idx = tid; idx < V2_TILE_M * V2_TILE_K; idx += V2_NTHREADS) {
                int rr = idx / V2_TILE_K;
                int kk = idx % V2_TILE_K;
                int v = row0 + rr;
                int s_idx = k0_n + kk;
                float val = 0.0f;
                if (v < n_variants && s_idx < n_samples) {
                    const unsigned char* row =
                        packed + (size_t)v * (size_t)bytes_per_variant;
                    val = volta2_z_of(row, s_idx, n_samples, mean[v], std_[v], count_a1);
                }
                sA_buf[nxt][rr * V2_SMEM_K_STRIDE + kk] = __float2half(val);
            }
            #pragma unroll
            for (int idx = tid; idx < V2_TILE_N * V2_TILE_K; idx += V2_NTHREADS) {
                int cc = idx / V2_TILE_K;
                int kk = idx % V2_TILE_K;
                int v = col0 + cc;
                int s_idx = k0_n + kk;
                float val = 0.0f;
                if (v < n_variants && s_idx < n_samples) {
                    const unsigned char* row =
                        packed + (size_t)v * (size_t)bytes_per_variant;
                    val = volta2_z_of(row, s_idx, n_samples, mean[v], std_[v], count_a1);
                }
                sB_buf[nxt][cc * V2_SMEM_K_STRIDE + kk] = __float2half(val);
            }
        }

        // ---- Compute on the CURRENT buffer ---------------------------------
        wmma::fragment<wmma::matrix_a, V2_WMMA_M, V2_WMMA_N, V2_WMMA_K,
                       __half, wmma::row_major> a_frag[V2_FRAGS_M];
        wmma::fragment<wmma::matrix_b, V2_WMMA_M, V2_WMMA_N, V2_WMMA_K,
                       __half, wmma::col_major> b_frag[V2_FRAGS_N];

        #pragma unroll
        for (int kp = 0; kp < V2_KSUB; ++kp) {
            int kp_off = kp * V2_WMMA_K;
            #pragma unroll
            for (int i = 0; i < V2_FRAGS_M; ++i) {
                const __half* a_ptr =
                    sA_buf[cur]
                    + (wy * V2_ROWS_PER_WARP + i * V2_WMMA_M) * V2_SMEM_K_STRIDE
                    + kp_off;
                wmma::load_matrix_sync(a_frag[i], a_ptr, V2_SMEM_K_STRIDE);
            }
            #pragma unroll
            for (int j = 0; j < V2_FRAGS_N; ++j) {
                const __half* b_ptr =
                    sB_buf[cur]
                    + (wx * V2_COLS_PER_WARP + j * V2_WMMA_N) * V2_SMEM_K_STRIDE
                    + kp_off;
                wmma::load_matrix_sync(b_frag[j], b_ptr, V2_SMEM_K_STRIDE);
            }
            #pragma unroll
            for (int i = 0; i < V2_FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < V2_FRAGS_N; ++j) {
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        // Ensure all threads finished consuming sA_buf[cur] before any thread
        // (during the NEXT iteration's prefetch) overwrites it. We also need
        // the prefetched stores into sA_buf[nxt] to be visible.
        __syncthreads();
    }

    // ---- Epilogue: store fragments to out with diagonal mirroring -----------
    #pragma unroll
    for (int i = 0; i < V2_FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < V2_FRAGS_N; ++j) {
            float* sptr = frag_scratch[warp_id];
            wmma::store_matrix_sync(sptr, acc[i][j], V2_WMMA_N, wmma::mem_row_major);
            #pragma unroll
            for (int e = lane_id; e < V2_WMMA_M * V2_WMMA_N; e += 32) {
                int rr_in = e / V2_WMMA_N;
                int cc_in = e % V2_WMMA_N;
                int rr = wy * V2_ROWS_PER_WARP + i * V2_WMMA_M + rr_in;
                int cc = wx * V2_COLS_PER_WARP + j * V2_WMMA_N + cc_in;
                int u = row0 + rr;
                int w = col0 + cc;
                if (u < n_variants && w < n_variants && rr < V2_TILE_M && cc < V2_TILE_N) {
                    float val = sptr[e];
                    if (u <= w) {
                        atomicAdd(&out[(size_t)u * (size_t)n_variants + w], val);
                        if (u != w) {
                            atomicAdd(&out[(size_t)w * (size_t)n_variants + u], val);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}
"""

# Dynamic shmem size required by the v2 kernel. Computed once at module load
# so the wrapper can both (a) opt the kernel into the 96 KB carveout via
# cudaFuncSetAttribute and (b) pass the correct ``shared_mem=`` byte count.
_V2_DYN_SHMEM_BYTES = 4 * 128 * 40 * 2  # 4 tiles * tile_m * (tile_k+8) * sizeof(half)


# ---------------------------------------------------------------------------
# Volta WMMA kernel (sm_70). Uses fp16 inputs + fp32 accumulator with the
# CUDA WMMA API (m16n16k16) for ~125 TFLOPS peak on V100. Compiles only
# on sm_70+ devices that support the API.
# ---------------------------------------------------------------------------

_GEMM_GRAM_VOLTA_SRC = r"""
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Re-declare the dosage decoder here (separate TU from the fp32 prelude).
__device__ __forceinline__ int volta_decode_dosage(int code, int count_a1) {
    if (code == 0x1) return -1;
    if (count_a1) {
        if (code == 0x0) return 2;
        if (code == 0x2) return 1;
        return 0;
    } else {
        if (code == 0x0) return 0;
        if (code == 0x2) return 1;
        return 2;
    }
}

// Compute fp32 standardized z for (variant v, sample i). Missing or
// zero-scale returns 0.0f.
__device__ __forceinline__ float volta_z_of(
    const unsigned char* __restrict__ row,
    int i,
    int n_samples,
    float m,
    float s,
    int count_a1)
{
    if (i >= n_samples) return 0.0f;
    int byte_idx = i >> 2;
    int slot = i & 0x3;
    int code = (row[byte_idx] >> (slot * 2)) & 0x3;
    int d = volta_decode_dosage(code, count_a1);
    if (d < 0) return 0.0f;
    if (!(s > 0.0f)) return 0.0f;
    return ((float)d - m) / s;
}

// Tile parameters: 128x128 output via 8x8 grid of 16x16 WMMA fragments.
// 8 warps (256 threads) arranged 4 row-warps x 2 col-warps:
//     warp_id = wy * VOLTA_WX + wx, with wy in 0..3, wx in 0..1.
// Each warp computes a 32x64 sub-tile = 2x4 WMMA fragments.
//
// k-tile = 32 samples per outer iteration, sub-divided into 2 WMMA k-steps
// of 16 each (VOLTA_KSUB=2). Per outer k-tile per warp:
//     2 a-frags * VOLTA_KSUB + 4 b-frags * VOLTA_KSUB loads, 2*4*VOLTA_KSUB=16 mma_sync.
//
// Static SMEM:
//   sA[VOLTA_TILE_M * VOLTA_SMEM_K_STRIDE]   = 128 * 40 * 2 B = 10240 B
//   sB[VOLTA_TILE_N * VOLTA_SMEM_K_STRIDE]   = 128 * 40 * 2 B = 10240 B
//   frag_scratch[VOLTA_NWARPS][16*16] float  =   8 * 256 * 4 B =  8192 B
//   Total ~28 KB — under the 48 KB per-block static SMEM limit on sm_70.
//
// Dynamic SMEM: ZERO. (launch.py gemm_gram_config(volta) returns shmem=0.)

#define VOLTA_TILE_M 128
#define VOLTA_TILE_N 128
#define VOLTA_TILE_K 32
#define VOLTA_WMMA_M 16
#define VOLTA_WMMA_N 16
#define VOLTA_WMMA_K 16
#define VOLTA_NWARPS 8
#define VOLTA_NTHREADS (VOLTA_NWARPS * 32)
#define VOLTA_WY 4  /* row-warps */
#define VOLTA_WX 2  /* col-warps */
#define VOLTA_ROWS_PER_WARP (VOLTA_TILE_M / VOLTA_WY)        /* 32 */
#define VOLTA_COLS_PER_WARP (VOLTA_TILE_N / VOLTA_WX)        /* 64 */
#define VOLTA_FRAGS_M (VOLTA_ROWS_PER_WARP / VOLTA_WMMA_M)   /*  2 */
#define VOLTA_FRAGS_N (VOLTA_COLS_PER_WARP / VOLTA_WMMA_N)   /*  4 */
#define VOLTA_KSUB (VOLTA_TILE_K / VOLTA_WMMA_K)             /*  2 */
// SMEM column stride includes a small skew to avoid bank conflicts on
// 16-element WMMA loads. Must be a multiple of 8 for fp16 loads.
#define VOLTA_SMEM_K_STRIDE (VOLTA_TILE_K + 8)

extern "C" __global__ void bitpacked_gemm_gram_volta_mma_kernel(
    const unsigned char* __restrict__ packed,
    int n_variants,
    int n_samples,
    int bytes_per_variant,
    const float* __restrict__ mean,
    const float* __restrict__ std_,
    float* __restrict__ out,
    int count_a1)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int row0 = bx * VOLTA_TILE_M;
    const int col0 = by * VOLTA_TILE_N;
    if (row0 >= n_variants || col0 >= n_variants) return;
    // Skip strictly-above-diagonal tiles.
    if (row0 > col0 + VOLTA_TILE_N - 1) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;                /* 0..7 */
    const int wy = warp_id / VOLTA_WX;           /* 0..3 */
    const int wx = warp_id % VOLTA_WX;           /* 0..1 */
    const int lane_id = tid & 31;

    // SMEM tiles (row-major, leading dim = VOLTA_SMEM_K_STRIDE).
    __shared__ __half sA[VOLTA_TILE_M * VOLTA_SMEM_K_STRIDE];
    __shared__ __half sB[VOLTA_TILE_N * VOLTA_SMEM_K_STRIDE];

    // Per-warp fragment accumulators: 2x4 grid of 16x16 fp32 frags.
    wmma::fragment<wmma::accumulator, VOLTA_WMMA_M, VOLTA_WMMA_N, VOLTA_WMMA_K, float>
        acc[VOLTA_FRAGS_M][VOLTA_FRAGS_N];
    #pragma unroll
    for (int i = 0; i < VOLTA_FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < VOLTA_FRAGS_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < n_samples; k0 += VOLTA_TILE_K) {
        // Cooperative load of sA[VOLTA_TILE_M][VOLTA_TILE_K].
        // 256 threads, 128*32 = 4096 elements -> 16 per thread.
        #pragma unroll
        for (int idx = tid; idx < VOLTA_TILE_M * VOLTA_TILE_K; idx += VOLTA_NTHREADS) {
            int rr = idx / VOLTA_TILE_K;
            int kk = idx % VOLTA_TILE_K;
            int v = row0 + rr;
            int s_idx = k0 + kk;
            float val = 0.0f;
            if (v < n_variants && s_idx < n_samples) {
                const unsigned char* row =
                    packed + (size_t)v * (size_t)bytes_per_variant;
                val = volta_z_of(row, s_idx, n_samples,
                                 mean[v], std_[v], count_a1);
            }
            sA[rr * VOLTA_SMEM_K_STRIDE + kk] = __float2half(val);
        }
        // Cooperative load of sB[VOLTA_TILE_N][VOLTA_TILE_K].
        #pragma unroll
        for (int idx = tid; idx < VOLTA_TILE_N * VOLTA_TILE_K; idx += VOLTA_NTHREADS) {
            int cc = idx / VOLTA_TILE_K;
            int kk = idx % VOLTA_TILE_K;
            int v = col0 + cc;
            int s_idx = k0 + kk;
            float val = 0.0f;
            if (v < n_variants && s_idx < n_samples) {
                const unsigned char* row =
                    packed + (size_t)v * (size_t)bytes_per_variant;
                val = volta_z_of(row, s_idx, n_samples,
                                 mean[v], std_[v], count_a1);
            }
            sB[cc * VOLTA_SMEM_K_STRIDE + kk] = __float2half(val);
        }
        __syncthreads();

        // Compute: iterate VOLTA_KSUB WMMA k-steps within the k-tile.
        wmma::fragment<wmma::matrix_a, VOLTA_WMMA_M, VOLTA_WMMA_N, VOLTA_WMMA_K,
                       __half, wmma::row_major> a_frag[VOLTA_FRAGS_M];
        wmma::fragment<wmma::matrix_b, VOLTA_WMMA_M, VOLTA_WMMA_N, VOLTA_WMMA_K,
                       __half, wmma::col_major> b_frag[VOLTA_FRAGS_N];

        #pragma unroll
        for (int kp = 0; kp < VOLTA_KSUB; ++kp) {
            int kp_off = kp * VOLTA_WMMA_K;
            // Load this warp's VOLTA_FRAGS_M a-frags (rows in [wy*32 .. wy*32+31]).
            #pragma unroll
            for (int i = 0; i < VOLTA_FRAGS_M; ++i) {
                const __half* a_ptr =
                    sA + (wy * VOLTA_ROWS_PER_WARP + i * VOLTA_WMMA_M)
                        * VOLTA_SMEM_K_STRIDE
                    + kp_off;
                wmma::load_matrix_sync(a_frag[i], a_ptr, VOLTA_SMEM_K_STRIDE);
            }
            // Load this warp's VOLTA_FRAGS_N b-frags (cols in [wx*64 .. wx*64+63]).
            // sB layout sB[cc*stride + kk] is col-major from the fragment POV
            // (kk fast, cc slow) — exactly what wmma::col_major expects.
            #pragma unroll
            for (int j = 0; j < VOLTA_FRAGS_N; ++j) {
                const __half* b_ptr =
                    sB + (wx * VOLTA_COLS_PER_WARP + j * VOLTA_WMMA_N)
                        * VOLTA_SMEM_K_STRIDE
                    + kp_off;
                wmma::load_matrix_sync(b_frag[j], b_ptr, VOLTA_SMEM_K_STRIDE);
            }
            #pragma unroll
            for (int i = 0; i < VOLTA_FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < VOLTA_FRAGS_N; ++j) {
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store each fragment to per-warp fp32 scratch then atomicAdd into out
    // with diagonal mirroring.
    __shared__ float frag_scratch[VOLTA_NWARPS][VOLTA_WMMA_M * VOLTA_WMMA_N];

    #pragma unroll
    for (int i = 0; i < VOLTA_FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < VOLTA_FRAGS_N; ++j) {
            float* sptr = frag_scratch[warp_id];
            wmma::store_matrix_sync(sptr, acc[i][j], VOLTA_WMMA_N,
                                    wmma::mem_row_major);
            // 32 lanes write 16*16 = 256 values; each lane handles 8.
            #pragma unroll
            for (int e = lane_id; e < VOLTA_WMMA_M * VOLTA_WMMA_N; e += 32) {
                int rr_in = e / VOLTA_WMMA_N;
                int cc_in = e % VOLTA_WMMA_N;
                int rr = wy * VOLTA_ROWS_PER_WARP + i * VOLTA_WMMA_M + rr_in;
                int cc = wx * VOLTA_COLS_PER_WARP + j * VOLTA_WMMA_N + cc_in;
                int u = row0 + rr;
                int w = col0 + cc;
                if (u < n_variants && w < n_variants && rr < VOLTA_TILE_M
                    && cc < VOLTA_TILE_N) {
                    float val = sptr[e];
                    if (u <= w) {
                        atomicAdd(&out[(size_t)u * (size_t)n_variants + w],
                                  val);
                        if (u != w) {
                            atomicAdd(
                                &out[(size_t)w * (size_t)n_variants + u],
                                val);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}
"""


# ---------------------------------------------------------------------------
# Lazy CuPy import + kernel compilation cache.
# ---------------------------------------------------------------------------


def _cupy() -> Any:
    import cupy as cp  # noqa: WPS433 — lazy import per spec

    return cp


@lru_cache(maxsize=8)
def _compiled_module_for(name: str) -> Any:
    """Compile a RawModule containing exactly ONE kernel entry point.

    This isolates per-arch shmem requirements: Volta (V100) need never compile
    the Ampere/Hopper variants whose static shmem exceeds the 48 KB per-block
    static limit. Only the arch-matched kernel is compiled at first use.
    """
    cp = _cupy()
    if name == _VOLTA_KERNEL_NAME:
        source = _GEMM_GRAM_VOLTA_SRC
    elif name == _VOLTA_V2_KERNEL_NAME:
        source = _GEMM_GRAM_VOLTA_V2_SRC
    elif name in _KERNEL_TAILS:
        source = _GEMM_GRAM_PRELUDE + _KERNEL_TAILS[name]
    else:
        raise KeyError(f"unknown gemm_gram kernel: {name}")
    module = cp.RawModule(
        code=source,
        options=("-std=c++14",),
        name_expressions=(name,),
    )
    return module


@lru_cache(maxsize=4)
def _get_kernel(name: str) -> Any:
    kernel = _compiled_module_for(name).get_function(name)
    if name == _VOLTA_V2_KERNEL_NAME:
        # Opt the kernel into Volta's 96 KB dynamic-shmem carveout. Without
        # this attribute the launch will fail with "too much shared data" since
        # the default per-block dynamic-shmem ceiling is 48 KB on sm_70.
        try:
            attr = 8  # cudaFuncAttributeMaxDynamicSharedMemorySize
            kernel.max_dynamic_shared_size_bytes = _V2_DYN_SHMEM_BYTES  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — fall back to attribute API
            try:
                cp = _cupy()
                cp.cuda.runtime.funcSetAttribute(
                    kernel.kernel.ptr, attr, _V2_DYN_SHMEM_BYTES
                )
            except Exception:  # noqa: BLE001 — caller will fall back to v1
                pass
    return kernel


def _kernel_for_arch(arch: Any) -> tuple[str, int, int]:
    """Return (kernel_name, tile_m, tile_n) for the given arch.

    ``arch`` may be a ``launch.GpuArch`` dataclass or a legacy string.

    Volta uses the v1 kernel: empirical V100 benchmarks at
    (97k samples, 512..8192 variants) show the double-buffered v2 kernel
    within 1-2% of v1 (both ~5 TFLOPS), so v1 is the production choice.
    """
    if arch == "hopper":
        return "bitpacked_gemm_gram_fp16_kernel", 128, 256
    if arch == "ampere" or arch == "ada":
        return "bitpacked_gemm_gram_tf32_kernel", 128, 128
    if arch == "t4" or arch == "turing":
        return "bitpacked_gemm_gram_dp4a_kernel", 64, 64
    if arch == "volta":
        return _VOLTA_KERNEL_NAME, 128, 128
    return "bitpacked_gemm_gram_fp32_kernel", 64, 64


# ---------------------------------------------------------------------------
# Public wrapper.
# ---------------------------------------------------------------------------


def gemm_gram(
    packed: Any,
    n_samples: int,
    mean: Any,
    std: Any,
    out: Any,
    count_a1: bool = True,
    stream: Any | None = None,
) -> None:
    """Accumulate B[u, v] += sum_i Z[i, u] * Z[i, v] into ``out``.

    Parameters
    ----------
    packed : cupy.ndarray (n_variants, bytes_per_variant) uint8
        Bitpacked variant-major matrix.
    n_samples : int
        Number of real samples (last 0–3 packed slots are padding).
    mean, std : cupy.ndarray (n_variants,) float32
    out : cupy.ndarray (n_variants, n_variants) float32, contiguous
        Accumulated INTO (not overwritten).
    count_a1 : bool, default True
        PLINK count_A1 convention selector.
    stream : cupy.cuda.Stream | None
        Launch stream; ``None`` -> current stream.
    """
    cp = _cupy()

    if packed.dtype != cp.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2D, got shape {packed.shape}")
    n_variants, bytes_per_variant = (int(packed.shape[0]), int(packed.shape[1]))

    if mean.dtype != cp.float32 or std.dtype != cp.float32:
        raise TypeError("mean and std must be float32")
    if mean.shape != (n_variants,) or std.shape != (n_variants,):
        raise ValueError("mean/std must have shape (n_variants,)")
    if out.dtype != cp.float32:
        raise TypeError(f"out must be float32, got {out.dtype}")
    if out.shape != (n_variants, n_variants):
        raise ValueError(
            f"out must be (n_variants, n_variants), got {out.shape}",
        )
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    if not packed.flags.c_contiguous:
        raise ValueError("packed must be C-contiguous")

    arch = _launch.gpu_arch()
    kernel_name, _tile_m, _tile_n = _kernel_for_arch(arch)
    cfg = _launch.gemm_gram_config(n_samples, n_variants, arch)

    try:
        kernel = _get_kernel(kernel_name)
    except Exception:  # pragma: no cover — fallback path
        # WMMA / tensor-core kernels may fail to compile on unexpected
        # toolchains; fall back to the always-safe fp32 path.
        if kernel_name != "bitpacked_gemm_gram_fp32_kernel":
            kernel_name = "bitpacked_gemm_gram_fp32_kernel"
            kernel = _get_kernel(kernel_name)
            cfg = _launch.gemm_gram_config(n_samples, n_variants, "unknown")
        else:
            raise

    args = (
        packed,
        cp.int32(n_variants),
        cp.int32(n_samples),
        cp.int32(bytes_per_variant),
        mean,
        std,
        out,
        cp.int32(1 if count_a1 else 0),
    )

    grid = cfg["grid"]
    block = cfg["block"]
    shmem = int(cfg.get("shmem_bytes", 0))
    if kernel_name == _VOLTA_V2_KERNEL_NAME:
        # v2 needs the 40 KB dynamic-shmem buffer for the double-buffered
        # SMEM tile pair (sA[2] + sB[2]); the per-warp fragment scratch is
        # still static.
        shmem = _V2_DYN_SHMEM_BYTES

    try:
        if stream is None:
            kernel(grid, block, args, shared_mem=shmem)
        else:
            with stream:
                kernel(grid, block, args, shared_mem=shmem)
    except Exception:  # noqa: BLE001 - v2 launch failure -> fall back to v1
        if kernel_name == _VOLTA_V2_KERNEL_NAME:
            kernel_name = _VOLTA_KERNEL_NAME
            kernel = _get_kernel(kernel_name)
            cfg = _launch.gemm_gram_config(n_samples, n_variants, "volta")
            shmem = int(cfg.get("shmem_bytes", 0))
            if stream is None:
                kernel(grid, block, args, shared_mem=shmem)
            else:
                with stream:
                    kernel(grid, block, args, shared_mem=shmem)
        else:
            raise


__all__ = ["gemm_gram"]

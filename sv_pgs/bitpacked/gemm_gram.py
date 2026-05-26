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
    # fp32 fallback (Volta / unknown): identical params to the dp4a tile.
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

_GEMM_GRAM_PRELUDE = _GEMM_GRAM_SRC


# ---------------------------------------------------------------------------
# Lazy CuPy import + kernel compilation cache.
# ---------------------------------------------------------------------------


def _cupy() -> Any:
    import cupy as cp  # noqa: WPS433 — lazy import per spec

    return cp


@lru_cache(maxsize=4)
def _compiled_module_for(name: str) -> Any:
    """Compile a RawModule containing exactly ONE kernel entry point.

    This isolates per-arch shmem requirements: Volta (V100) need never compile
    the Ampere/Hopper variants whose static shmem exceeds the 48 KB per-block
    static limit. Only the arch-matched kernel is compiled at first use.
    """
    cp = _cupy()
    if name not in _KERNEL_TAILS:
        raise KeyError(f"unknown gemm_gram kernel: {name}")
    source = _GEMM_GRAM_PRELUDE + _KERNEL_TAILS[name]
    module = cp.RawModule(
        code=source,
        options=("-std=c++14",),
        name_expressions=(name,),
    )
    return module


@lru_cache(maxsize=4)
def _get_kernel(name: str) -> Any:
    return _compiled_module_for(name).get_function(name)


def _kernel_for_arch(arch: str) -> tuple[str, int, int]:
    """Return (kernel_name, tile_m, tile_n) for the given arch."""
    if arch == "hopper":
        return "bitpacked_gemm_gram_fp16_kernel", 128, 256
    if arch == "ampere":
        return "bitpacked_gemm_gram_tf32_kernel", 128, 128
    if arch == "t4":
        return "bitpacked_gemm_gram_dp4a_kernel", 64, 64
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

    kernel = _get_kernel(kernel_name)

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

    if stream is None:
        kernel(grid, block, args, shared_mem=shmem)
    else:
        with stream:
            kernel(grid, block, args, shared_mem=shmem)


__all__ = ["gemm_gram"]

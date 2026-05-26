from __future__ import annotations

import functools
from typing import Any

from sv_pgs.bitpacked.launch import gemv_nt_config, gpu_arch

# Variant-chunk size: number of variants whose (x, mean, scale) we stage in
# shared memory per inner pass. Must be <= block_x. 64 keeps SMEM tiny
# (64 * 3 * 4 = 768 B) and amortizes the cooperative load.
_CHUNK_V = 64

_CUDA_SOURCE = r"""
extern "C" {

#ifndef CHUNK_V
#define CHUNK_V 64
#endif

__global__ void bitpacked_gemv_nt_kernel(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ scale,
    float* __restrict__ out,
    const int n_samples,
    const int n_variants,
    const int bytes_per_variant,
    const int count_a1)
{
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    const int byte_idx = sample >> 2;
    const int slot     = sample & 3;
    const int shift    = slot << 1;     // 0,2,4,6
    const bool active  = (sample < n_samples);

    __shared__ float s_x[CHUNK_V];
    __shared__ float s_m[CHUNK_V];
    __shared__ float s_inv[CHUNK_V];   // 1/scale, or 0 if scale<=0/NaN -> zero contribution

    float acc = 0.0f;

    for (int v0 = 0; v0 < n_variants; v0 += CHUNK_V) {
        const int chunk = (v0 + CHUNK_V <= n_variants) ? CHUNK_V : (n_variants - v0);

        if (tid < chunk) {
            const float sc = scale[v0 + tid];
            s_inv[tid] = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
            s_x[tid]   = x[v0 + tid];
            s_m[tid]   = mean[v0 + tid];
        }
        __syncthreads();

        if (active) {
            #pragma unroll 4
            for (int k = 0; k < chunk; ++k) {
                const int v = v0 + k;
                const unsigned int b = (unsigned int)__ldg(packed + (size_t)v * (size_t)bytes_per_variant + byte_idx);
                const unsigned int code = (b >> shift) & 0x3u;
                // Inline decode (no LUT -> no __constant__ serialization across warp):
                //   base = (code>>1) + (code&1)  in {0,1,1,2} for codes {00,01,10,11}
                //   count_a1=True : dose = 2 - base   -> {2, 1, 1, 0}
                //   count_a1=False: dose = base       -> {0, 1, 1, 2}
                // Both correctly map non-missing codes. For code==1 (missing) we
                // would get dose=1 either way; we mask out below.
                const int base = (int)((code >> 1) + (code & 1u));
                const int dose = count_a1 ? (2 - base) : base;
                const float miss_mask = (code == 1u) ? 0.0f : 1.0f;
                const float raw = (float)dose;
                const float z = (raw - s_m[k]) * s_inv[k] * miss_mask;
                acc += z * s_x[k];
            }
        }
        __syncthreads();
    }

    if (active) {
        out[sample] += acc;
    }
}

}  // extern "C"
"""


@functools.lru_cache(maxsize=4)
def _get_kernel(_key: tuple[Any, ...]) -> Any:
    import cupy as cp

    src = _CUDA_SOURCE.replace("#define CHUNK_V 64", f"#define CHUNK_V {_CHUNK_V}", 1)
    module = cp.RawModule(code=src, options=("--std=c++14", "-use_fast_math"))
    kernel = module.get_function("bitpacked_gemv_nt_kernel")
    return kernel


def gemv_nt(
    packed: "Any",
    n_samples: int,
    x: "Any",
    mean: "Any",
    std: "Any",
    out: "Any",
    count_a1: bool = True,
    stream: "Any | None" = None,
) -> None:
    """Compute y[i] += sum_v ((dosage[i,v] - mean[v]) / std[v]) * x[v]; missing -> 0.

    `out` must be initialized by the caller (e.g. cupy.zeros). Kernel adds INTO out.
    """
    import cupy as cp

    if packed.ndim != 2 or packed.dtype != cp.uint8:
        raise ValueError("packed must be 2D uint8 (n_variants, bytes_per_variant)")
    n_variants, bytes_per_variant = packed.shape
    expected_bpv = (n_samples + 3) // 4
    if bytes_per_variant != expected_bpv:
        raise ValueError(
            f"bytes_per_variant {bytes_per_variant} != ceil(n_samples/4)={expected_bpv}"
        )
    for name, arr, n in (("x", x, n_variants), ("mean", mean, n_variants), ("std", std, n_variants)):
        if arr.ndim != 1 or arr.shape[0] != n or arr.dtype != cp.float32:
            raise ValueError(f"{name} must be float32 of length {n}")
    if out.ndim != 1 or out.shape[0] != n_samples or out.dtype != cp.float32:
        raise ValueError(f"out must be float32 of length {n_samples}")
    if not packed.flags.c_contiguous:
        raise ValueError("packed must be C-contiguous")

    arch = gpu_arch()
    cfg = gemv_nt_config(n_samples, n_variants, arch)

    block_x = int(cfg["block"][0])
    if block_x < _CHUNK_V:
        raise RuntimeError(f"block_x ({block_x}) must be >= CHUNK_V ({_CHUNK_V})")

    kernel = _get_kernel((arch, cfg["block"], _CHUNK_V))

    args = (
        packed.data.ptr,
        x.data.ptr,
        mean.data.ptr,
        std.data.ptr,
        out.data.ptr,
        cp.int32(n_samples),
        cp.int32(n_variants),
        cp.int32(bytes_per_variant),
        cp.int32(1 if count_a1 else 0),
    )

    if stream is None:
        kernel(cfg["grid"], cfg["block"], args)
    else:
        with stream:
            kernel(cfg["grid"], cfg["block"], args)

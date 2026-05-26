"""Transpose bitpacked GEMV: g[v] += sum_i Z[i,v] * y[i].

One CUDA block owns exactly one variant (a "variant tile" of size 1). The
config in ``sv_pgs.bitpacked.launch.gemv_tn_config`` returns
``grid = (n_variants, 1, 1)``, so each block computes one entry of the output
vector. The block walks the sample axis in chunks of ``blockDim.x``,
cooperatively loads ``y`` into shared memory, unpacks the packed bytes for
its variant via a constant-memory LUT, accumulates standardized contributions
in per-thread registers, then performs an in-block warp-shuffle + shared-mem
reduction. Because each variant is owned by exactly one block, the final
write to ``out`` is a plain ``+=`` without atomics.
"""

from __future__ import annotations

from typing import Any

from sv_pgs.bitpacked.launch import gemv_tn_config, gpu_arch

_KERNEL_NAME = "bitpacked_gemv_tn_kernel"

_KERNEL_SOURCE = r"""
extern "C" {

// 256 x 4 decode LUT. Layout: [byte * 4 + slot] -> int8 dosage (or -127 missing).
// Filled from host before each launch via cudaMemcpyToSymbol equivalent
// (cupy RawKernel handles __constant__ via a separate memcpy; we use a plain
// device-global array initialized from the host on first use through a
// dedicated init kernel below). To keep this self-contained and avoid host-
// side symbol manipulation, we instead encode the decode logic inline.
//
// The 2-bit codes per PLINK 1.9 (count_A1 = true):
//   0b00 -> dosage 2
//   0b01 -> missing
//   0b10 -> dosage 1
//   0b11 -> dosage 0
// count_A1 = false swaps 0b00 <-> 0b11.

__device__ __forceinline__ void decode_slot(
    unsigned int code,
    int count_a1,
    int* out_dosage,
    int* out_missing)
{
    int miss = (code == 1u) ? 1 : 0;
    int dose;
    if (count_a1) {
        // 00->2, 01->miss, 10->1, 11->0
        // mapping: 0->2, 2->1, 3->0
        if (code == 0u)      dose = 2;
        else if (code == 2u) dose = 1;
        else                 dose = 0; // covers code==3 and code==1 (will be masked)
    } else {
        // 00->0, 01->miss, 10->1, 11->2
        if (code == 0u)      dose = 0;
        else if (code == 2u) dose = 1;
        else                 dose = 2; // covers code==3 and code==1 (will be masked)
    }
    *out_dosage = dose;
    *out_missing = miss;
}

__global__ void bitpacked_gemv_tn_kernel(
    const unsigned char* __restrict__ packed,   // (n_variants, bytes_per_variant)
    const float* __restrict__ y,                // (n_samples,)
    const float* __restrict__ mean,             // (n_variants,)
    const float* __restrict__ std,              // (n_variants,)
    float* __restrict__ out,                    // (n_variants,) accumulated INTO
    int n_samples,
    int n_variants,
    int bytes_per_variant,
    int count_a1)
{
    const int v = blockIdx.x;
    if (v >= n_variants) return;

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    extern __shared__ float smem[];
    float* y_tile = smem;                       // size = bdim floats
    float* red    = smem + bdim;                // size = (bdim/32) floats for warp reduction

    const float mv = mean[v];
    const float sv = std[v];
    const float inv_s = (sv > 0.0f) ? (1.0f / sv) : 0.0f;

    const unsigned char* row = packed + (size_t)v * (size_t)bytes_per_variant;

    float acc = 0.0f;

    // Walk the sample axis in chunks of bdim samples.
    for (int s0 = 0; s0 < n_samples; s0 += bdim) {
        const int s_local_count = (s0 + bdim <= n_samples) ? bdim : (n_samples - s0);

        // Cooperatively load y[s0 : s0 + s_local_count] into shared memory.
        if (tid < s_local_count) {
            y_tile[tid] = y[s0 + tid];
        }
        __syncthreads();

        // Each thread handles its own sample index within the chunk.
        if (tid < s_local_count) {
            const int sample_idx = s0 + tid;
            const int byte_idx = sample_idx >> 2;
            const int slot    = sample_idx & 3;
            const unsigned int byte_val = (unsigned int)row[byte_idx];
            const unsigned int code = (byte_val >> (2 * slot)) & 0x3u;

            int dose, miss;
            decode_slot(code, count_a1, &dose, &miss);

            if (!miss) {
                const float raw = (float)dose;
                const float z = (raw - mv) * inv_s;
                acc += z * y_tile[tid];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction via shuffles.
    unsigned int mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(mask, acc, offset);
    }

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (bdim + 31) >> 5;

    if (lane == 0) {
        red[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v_partial = (tid < n_warps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v_partial += __shfl_xor_sync(mask, v_partial, offset);
        }
        if (tid == 0) {
            out[v] += v_partial;
        }
    }
}

} // extern "C"
"""


_kernel_cache: dict[str, Any] = {}


def _get_kernel() -> Any:
    if "k" in _kernel_cache:
        return _kernel_cache["k"]
    import cupy as cp  # lazy

    kernel = cp.RawKernel(_KERNEL_SOURCE, _KERNEL_NAME)
    _kernel_cache["k"] = kernel
    return kernel


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
        raise ValueError(f"packed must be 2-D (n_variants, bytes_per_variant), got shape {packed.shape}")

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

    if n_variants == 0 or n_samples == 0:
        return

    arch = gpu_arch()
    cfg = gemv_tn_config(int(n_samples), int(n_variants), arch)
    grid = cfg["grid"]
    block = cfg["block"]

    block_x = int(block[0])
    # Shared memory: bdim floats for the y tile + n_warps floats for the
    # warp-reduction scratch.
    n_warps = (block_x + 31) // 32
    shmem_bytes = (block_x + n_warps) * 4

    kernel = _get_kernel()

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

"""Transpose bitpacked GEMV: g[v] += sum_i Z[i,v] * y[i].

This kernel uses a *variant-tile* design: each CUDA block owns ``VPB``
contiguous variants (default 32) instead of just one. Inside a block, the
``BLOCK = VPB * TPV`` threads are split into ``VPB`` variant-groups of
``TPV`` threads. Each group walks the sample axis cooperatively in chunks
of ``CHUNK_SAMPLES`` samples (default 256 = TPV * SAMPLES_PER_THREAD), with
each thread issuing 4-byte (uint32) loads to amortize HBM access (4 bytes
= 16 samples per load). The ``y`` tile is cooperatively staged into shared
memory once per chunk and re-used by all 32 variants in the block. Per-
group partials are reduced via 2 ``__shfl_xor_sync`` rounds (TPV = 4), and
the leader of each group writes ``out[v] += partial`` (no atomics — each
variant has exactly one block of writers).

Why this beats the one-block-per-variant design:
- 32x fewer blocks → much lower scheduling / launch overhead per variant.
- ``y`` reads are amortized 32x across variants in the same block (shared
  memory tile + L1 hits across variant rows).
- 4 threads per variant gives a 4x sample-axis parallel reduction without
  inter-warp synchronization (the 4 threads always share a warp).
- Decode uses the same ``__constant__`` LUT layout as ``gemv_nt``.
"""

from __future__ import annotations

import functools
from typing import Any

from sv_pgs.bitpacked.launch import gemv_tn_config, gpu_arch

_KERNEL_NAME = "bitpacked_gemv_tn_kernel"

_PLINK_MISSING_INT8 = -127

# Tile parameters baked into the kernel. Keep in sync with the CUDA source.
# VPB     : variants per block
# TPV     : threads per variant (must divide warp size 32)
# SPT     : sample bytes per thread per inner iter (multiple of 4 for u32 loads)
# CHUNK_S : samples processed by the block per outer iter = TPV * SPT * 4
_VPB = 32
_TPV = 4
_SPT_BYTES = 16  # 16 bytes = 64 samples per thread per inner unrolled iter
_CHUNK_S = _TPV * _SPT_BYTES * 4  # 256 samples per outer iter

_CUDA_SOURCE = r"""
extern "C" {

__constant__ signed char BITPACKED_DECODE_LUT[256 * 8];

#ifndef VPB
#define VPB 32
#endif
#ifndef TPV
#define TPV 4
#endif
#ifndef SPT_BYTES
#define SPT_BYTES 16
#endif

// CHUNK_S = TPV * SPT_BYTES * 4  -- samples processed per block per outer iter.
#define CHUNK_S (TPV * SPT_BYTES * 4)
#define BLOCK (VPB * TPV)

__global__ void bitpacked_gemv_tn_kernel(
    const unsigned char* __restrict__ packed,   // (n_variants, bytes_per_variant)
    const float* __restrict__ y,                // (n_samples,)
    const float* __restrict__ mean,             // (n_variants,)
    const float* __restrict__ std,              // (n_variants,)
    float* __restrict__ out,                    // (n_variants,) accumulated INTO
    const int n_samples,
    const int n_variants,
    const int bytes_per_variant,
    const int count_a1)
{
    // Variant assigned to this thread (TPV threads share one variant).
    const int tid = threadIdx.x;
    const int v_local = tid / TPV;       // 0..VPB-1
    const int sub     = tid % TPV;       // 0..TPV-1
    const int v_global = blockIdx.x * VPB + v_local;

    const int lut_off = count_a1 ? 0 : 1;

    // Shared y-tile (CHUNK_S floats) -- cooperatively loaded each outer iter.
    extern __shared__ float smem[];
    float* y_tile = smem;  // size = CHUNK_S floats

    // Per-variant constants (broadcast within the variant-group; threads in
    // the same group all load the same value -- fine, it's a single fp32 op).
    float mv = 0.0f;
    float inv_s = 0.0f;
    if (v_global < n_variants) {
        mv = mean[v_global];
        const float sv = std[v_global];
        // gate scale > 0 (also rejects NaN); zero-std columns contribute 0.
        inv_s = (sv > 0.0f) ? (__frcp_rn(sv)) : 0.0f;
    }

    // Row base offset (in bytes) for this variant.
    const size_t row_off = (size_t)v_global * (size_t)bytes_per_variant;

    float acc = 0.0f;

    // Walk samples in CHUNK_S-sized blocks.
    for (int s0 = 0; s0 < n_samples; s0 += CHUNK_S) {
        const int chunk = (s0 + CHUNK_S <= n_samples) ? CHUNK_S : (n_samples - s0);

        // Cooperatively load y[s0 : s0+chunk] into shared memory.
        // BLOCK threads, CHUNK_S floats -> CHUNK_S / BLOCK floats per thread.
        #pragma unroll
        for (int k = tid; k < CHUNK_S; k += BLOCK) {
            const int s = s0 + k;
            y_tile[k] = (s < n_samples) ? y[s] : 0.0f;
        }
        __syncthreads();

        if (v_global < n_variants) {
            // This thread's byte slab within the chunk:
            //   bytes [sub*SPT_BYTES, sub*SPT_BYTES + SPT_BYTES)
            //   covers samples [sub*SPT_BYTES*4, (sub+1)*SPT_BYTES*4)
            const int byte_base_in_chunk = sub * SPT_BYTES;
            const int sample_base_in_chunk = byte_base_in_chunk * 4;
            const int global_byte_base = (s0 >> 2) + byte_base_in_chunk;

            // Read SPT_BYTES bytes for this variant. Use uint32 loads where
            // alignment + range allow it (4 bytes = 16 samples per load).
            // Each thread reads SPT_BYTES bytes = SPT_BYTES/4 uint32s.
            // We guard the tail (last chunk may be partial).
            const unsigned char* row_bytes = packed + row_off + global_byte_base;

            #pragma unroll
            for (int w = 0; w < SPT_BYTES; w += 4) {
                // Local sample range covered by this 4-byte group:
                //   samples [sample_base_in_chunk + w*4, sample_base_in_chunk + w*4 + 16)
                const int s_local0 = sample_base_in_chunk + w * 4;

                // Skip entirely if past chunk boundary.
                if (s_local0 >= chunk) break;

                // Load 4 packed bytes. Use __ldg for byte loads; the compiler
                // emits 4 individual LDG.E.U8 instructions which the memory
                // pipeline coalesces along the warp.
                const int gbb = global_byte_base + w;
                unsigned int b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                if (gbb + 0 < bytes_per_variant) b0 = (unsigned int)__ldg(row_bytes + w + 0);
                if (gbb + 1 < bytes_per_variant) b1 = (unsigned int)__ldg(row_bytes + w + 1);
                if (gbb + 2 < bytes_per_variant) b2 = (unsigned int)__ldg(row_bytes + w + 2);
                if (gbb + 3 < bytes_per_variant) b3 = (unsigned int)__ldg(row_bytes + w + 3);

                // Decode 16 samples (4 bytes x 4 slots) via the constant LUT.
                // BITPACKED_DECODE_LUT layout: [byte*8 + slot*2 + lut_off].
                // Each fused contribution: missing -> 0; else (d - mv) * inv_s * y[i].
                #pragma unroll
                for (int bidx = 0; bidx < 4; ++bidx) {
                    const int byte_val = (bidx == 0) ? (int)b0
                                       : (bidx == 1) ? (int)b1
                                       : (bidx == 2) ? (int)b2 : (int)b3;
                    const int s_local_byte = s_local0 + bidx * 4;
                    #pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const int s_local = s_local_byte + slot;
                        if (s_local >= chunk) break;
                        const signed char d = BITPACKED_DECODE_LUT[(byte_val << 3) + (slot << 1) + lut_off];
                        if (d != (signed char)-127) {
                            const float z = ((float)d - mv) * inv_s;
                            acc += z * y_tile[s_local];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // Intra-variant reduction across TPV threads (all in same warp by
    // construction: VPB * TPV = 128, TPV = 4, so each variant's 4 threads
    // lie in consecutive lanes of one warp).
    // We need to reduce across lanes (lane & (TPV-1)).
    #pragma unroll
    for (int offset = TPV >> 1; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, offset);
    }

    if (sub == 0 && v_global < n_variants) {
        out[v_global] += acc;
    }
}

}  // extern "C"
"""


def _build_decode_lut_bytes() -> bytes:
    """Build the 256x8 int8 LUT (same layout as ``gemv_nt``).

    Layout: ``table[byte * 8 + slot * 2 + lut_off]`` -> dosage in {-127, 0, 1, 2}.
    ``lut_off = 0`` for count_a1, ``lut_off = 1`` for count_a2.
    """
    mapping_a1 = {0b00: 2, 0b01: _PLINK_MISSING_INT8, 0b10: 1, 0b11: 0}
    mapping_a2 = {0b00: 0, 0b01: _PLINK_MISSING_INT8, 0b10: 1, 0b11: 2}
    table = bytearray(256 * 8)
    for byte in range(256):
        for slot in range(4):
            code = (byte >> (slot * 2)) & 0b11
            v_a1 = mapping_a1[code] & 0xFF
            v_a2 = mapping_a2[code] & 0xFF
            table[byte * 8 + slot * 2 + 0] = v_a1
            table[byte * 8 + slot * 2 + 1] = v_a2
    return bytes(table)


@functools.lru_cache(maxsize=4)
def _get_kernel(_key: tuple[Any, ...]) -> Any:
    import numpy as np

    import cupy as cp

    options = (
        "--std=c++14",
        f"-DVPB={_VPB}",
        f"-DTPV={_TPV}",
        f"-DSPT_BYTES={_SPT_BYTES}",
    )
    module = cp.RawModule(code=_CUDA_SOURCE, options=options)
    kernel = module.get_function("bitpacked_gemv_tn_kernel")
    lut_bytes = _build_decode_lut_bytes()
    lut_host_np = np.frombuffer(lut_bytes, dtype=np.int8)
    lut_dev = cp.asarray(lut_host_np)
    const_ptr = module.get_global("BITPACKED_DECODE_LUT")
    dst_ptr = const_ptr.ptr if hasattr(const_ptr, "ptr") else int(const_ptr)
    cp.cuda.runtime.memcpy(
        int(dst_ptr),
        int(lut_dev.data.ptr),
        lut_dev.nbytes,
        cp.cuda.runtime.memcpyDeviceToDevice,
    )
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

    # Probe arch (still consulted for future per-arch tuning, but the kernel
    # itself is tile-size-fixed via -D macros at compile time).
    _ = gemv_tn_config(int(n_samples), int(n_variants), gpu_arch())

    block_x = _VPB * _TPV  # 128
    grid_x = (n_variants + _VPB - 1) // _VPB
    grid = (grid_x, 1, 1)
    block = (block_x, 1, 1)
    shmem_bytes = _CHUNK_S * 4  # CHUNK_S floats for the y tile

    kernel = _get_kernel(("v1", _VPB, _TPV, _SPT_BYTES))

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

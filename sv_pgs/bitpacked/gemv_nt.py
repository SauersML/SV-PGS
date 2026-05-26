from __future__ import annotations

import functools
from typing import Any

from sv_pgs.bitpacked.launch import gemv_nt_config, gpu_arch

_PLINK_MISSING_INT8 = -127

_CUDA_SOURCE = r"""
extern "C" {

__constant__ signed char BITPACKED_DECODE_LUT[256 * 8];

__global__ void bitpacked_gemv_nt_kernel(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    float* __restrict__ out,
    const int n_samples,
    const int n_variants,
    const int bytes_per_variant,
    const int count_a1)
{
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    const int lut_off = count_a1 ? 0 : 1;

    if (sample >= n_samples) return;

    const int byte_idx = sample >> 2;          // sample / 4
    const int slot = sample & 3;               // sample % 4
    const int slot_off = (slot << 1) + lut_off;

    float acc = 0.0f;

    int v = 0;
    // unrolled by 4 over variants
    for (; v + 4 <= n_variants; v += 4) {
        const unsigned char b0 = __ldg(packed + (v + 0) * bytes_per_variant + byte_idx);
        const unsigned char b1 = __ldg(packed + (v + 1) * bytes_per_variant + byte_idx);
        const unsigned char b2 = __ldg(packed + (v + 2) * bytes_per_variant + byte_idx);
        const unsigned char b3 = __ldg(packed + (v + 3) * bytes_per_variant + byte_idx);

        const signed char d0 = BITPACKED_DECODE_LUT[((int)b0 << 3) + slot_off];
        const signed char d1 = BITPACKED_DECODE_LUT[((int)b1 << 3) + slot_off];
        const signed char d2 = BITPACKED_DECODE_LUT[((int)b2 << 3) + slot_off];
        const signed char d3 = BITPACKED_DECODE_LUT[((int)b3 << 3) + slot_off];

        // fuse: missing(-127) -> 0; else (dosage - mean) * (x / std)
        const float m0 = mean[v + 0], m1 = mean[v + 1], m2 = mean[v + 2], m3 = mean[v + 3];
        const float s0 = std[v + 0],  s1 = std[v + 1],  s2 = std[v + 2],  s3 = std[v + 3];
        const float x0 = x[v + 0],    x1 = x[v + 1],    x2 = x[v + 2],    x3 = x[v + 3];

        // gate scale > 0 (also rejects NaN); zero-std columns contribute 0
        const float z0 = ((d0 == (signed char)-127) || !(s0 > 0.0f)) ? 0.0f : (((float)d0 - m0) / s0) * x0;
        const float z1 = ((d1 == (signed char)-127) || !(s1 > 0.0f)) ? 0.0f : (((float)d1 - m1) / s1) * x1;
        const float z2 = ((d2 == (signed char)-127) || !(s2 > 0.0f)) ? 0.0f : (((float)d2 - m2) / s2) * x2;
        const float z3 = ((d3 == (signed char)-127) || !(s3 > 0.0f)) ? 0.0f : (((float)d3 - m3) / s3) * x3;

        acc += z0 + z1 + z2 + z3;
    }
    for (; v < n_variants; ++v) {
        const unsigned char b = __ldg(packed + v * bytes_per_variant + byte_idx);
        const signed char d = BITPACKED_DECODE_LUT[((int)b << 3) + slot_off];
        const float s = std[v];
        if (d != (signed char)-127 && (s > 0.0f)) {
            acc += (((float)d - mean[v]) / s) * x[v];
        }
    }

    atomicAdd(out + sample, acc);
}

}  // extern "C"
"""


def _build_decode_lut_bytes() -> bytes:
    # Build 256 x 8 int8 table: [byte][slot*2 + count_a1_bit]
    # count_a1=True -> bit 0; count_a1=False -> bit 1
    # count_a1=True: 0b00->2, 0b01->missing, 0b10->1, 0b11->0
    # count_a1=False: 0b00->0, 0b01->missing, 0b10->1, 0b11->2
    mapping_a1 = {0b00: 2, 0b01: _PLINK_MISSING_INT8, 0b10: 1, 0b11: 0}
    mapping_a2 = {0b00: 0, 0b01: _PLINK_MISSING_INT8, 0b10: 1, 0b11: 2}
    table = bytearray(256 * 8)
    for byte in range(256):
        for slot in range(4):
            code = (byte >> (slot * 2)) & 0b11
            v_a1 = mapping_a1[code] & 0xFF  # two's complement int8
            v_a2 = mapping_a2[code] & 0xFF
            table[byte * 8 + slot * 2 + 0] = v_a1
            table[byte * 8 + slot * 2 + 1] = v_a2
    return bytes(table)


@functools.lru_cache(maxsize=4)
def _get_kernel(_key: tuple[Any, ...]) -> Any:
    import numpy as np

    import cupy as cp

    module = cp.RawModule(code=_CUDA_SOURCE, options=("--std=c++14",))
    kernel = module.get_function("bitpacked_gemv_nt_kernel")
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

    kernel = _get_kernel((arch, cfg["block"], cfg["grid"][0] > 0))

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
        kernel(cfg["grid"], cfg["block"], args, shared_mem=cfg.get("shmem_bytes", 0))
    else:
        with stream:
            kernel(cfg["grid"], cfg["block"], args, shared_mem=cfg.get("shmem_bytes", 0))

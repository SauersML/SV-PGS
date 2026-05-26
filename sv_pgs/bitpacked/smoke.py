"""End-to-end bitpacked GPU pipeline smoke check.

Builds a tiny synthetic PLINK 1.9 BED (50 samples x 200 variants with missing
slots), then exercises the four production-critical layers in order:

  1. Environment:    cupy import + libnvrtc dlopen via a trivial RawKernel.
  2. Screening:      ``screening_pipeline.run_screening_pass`` end-to-end,
                     parity-checked against ``cpu_reference.cpu_screen``.
  3. Matrix build:   ``BitpackedDeviceMatrix`` constructor.
  4. GEMV / Gram:    ``gemv_nt`` / ``gemv_tn`` / ``gemm_gram`` parity-checked
                     against the CPU reference.

Designed as the FIRST thing ``run.sh`` runs after ``uv sync`` — runs in a
few seconds and catches all four classes of breakage that production has hit
(libnvrtc missing, kernel compile error, kernel correctness regression,
screening_pipeline integration bug) BEFORE a 90-minute fit kicks off.

Prints ``BITPACKED PIPELINE OK`` and exits 0 on success; prints the failing
stage + exception type + message and exits non-zero otherwise.

This module is GPU-only by design and imports cupy unconditionally.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable


def _say(msg: str) -> None:
    print(msg, flush=True)


def _stage(name: str, fn: Callable[[], None]) -> None:
    _say(f"  [stage] {name} ...")
    fn()
    _say(f"  [ok]    {name}")


def _make_tiny_bed(bed_path: Path, n_samples: int, n_variants: int, seed: int) -> "object":
    """Write a synthetic PLINK 1.9 BED with intentional missing values.

    Returns the (n_samples, n_variants) float32 dosage matrix (with NaN at
    missing slots) so the smoke can also feed it through the CPU reference.
    """
    import numpy as np

    from sv_pgs.plink import to_bed

    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, size=(n_samples, n_variants), dtype=np.int8)
    dosage = np.where(
        codes == 0,
        2.0,
        np.where(codes == 1, np.nan, np.where(codes == 2, 1.0, 0.0)),
    ).astype(np.float32)
    properties = {
        "iid": [f"s{i}" for i in range(n_samples)],
        "sid": [f"v{j}" for j in range(n_variants)],
    }
    to_bed(bed_path, dosage, properties)
    return dosage


def main() -> int:
    n_samples = 50
    n_variants = 200
    seed = 1729

    # Stage 0: cupy import + nvrtc compile probe. This is the fastest way to
    # surface the AoU libnvrtc.so.12 failure.
    try:
        _stage(
            "cupy import + libnvrtc dlopen (RawKernel compile)",
            _stage0_nvrtc,
        )
    except BaseException as exc:  # noqa: BLE001
        _say(f"  [FAIL]  stage 0 (nvrtc): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        _say(
            "  hint: prepend .venv/lib/python3.*/site-packages/nvidia/*/lib to "
            "LD_LIBRARY_PATH (run.sh now does this automatically)."
        )
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "smoke.bed"
        try:
            dosage = _make_tiny_bed(bed_path, n_samples, n_variants, seed)
        except BaseException as exc:  # noqa: BLE001
            _say(f"  [FAIL]  synthetic BED creation: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 3

        try:
            _stage(
                "run_screening_pass parity vs cpu_screen",
                lambda: _stage_screening(bed_path, n_samples, n_variants),
            )
        except BaseException as exc:  # noqa: BLE001
            _say(f"  [FAIL]  screening: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 4

        try:
            _stage(
                "BitpackedDeviceMatrix + gemv_nt / gemv_tn / gemm_gram parity",
                lambda: _stage_matrix_kernels(bed_path, n_samples, n_variants, dosage),
            )
        except BaseException as exc:  # noqa: BLE001
            _say(f"  [FAIL]  matrix kernels: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 5

    _say("BITPACKED PIPELINE OK")
    return 0


def _stage0_nvrtc() -> None:
    import cupy as cp

    kernel = cp.RawKernel(
        r'extern "C" __global__ void __sv_pgs_smoke_noop() {}',
        "__sv_pgs_smoke_noop",
    )
    kernel((1,), (1,), ())
    cp.cuda.Stream.null.synchronize()


def _stage_screening(bed_path: Path, n_samples: int, n_variants: int) -> None:
    import numpy as np

    from sv_pgs.bitpacked.cpu_reference import cpu_screen
    from sv_pgs.mmap_reader import BedMmapReader
    from sv_pgs.screening_pipeline import run_screening_pass

    # GPU path through the production code.
    res = run_screening_pass(
        [bed_path],
        [n_samples],
        [n_variants],
        sample_intersect=None,
        rhs=None,
        count_a1=True,
    )

    # CPU reference straight off the packed bytes.
    with BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants) as reader:
        packed = np.array(reader.read_all_packed(), copy=True)
    ref = cpu_screen(packed, n_samples=n_samples, count_a1=True)

    np.testing.assert_array_equal(res["count"], ref["count"])
    np.testing.assert_allclose(res["sum"], ref["sum"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(res["sumsq"], ref["sumsq"], rtol=0, atol=1e-9)


def _stage_matrix_kernels(
    bed_path: Path,
    n_samples: int,
    n_variants: int,
    dosage: "object",  # numpy array; typed loosely to avoid mod-level numpy import
) -> None:
    import cupy as cp
    import numpy as np

    from sv_pgs.bitpacked.cpu_reference import (
        compute_mean_scale,
        cpu_gemm_gram,
        cpu_gemv_nt,
        cpu_gemv_tn,
    )
    from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix
    from sv_pgs.mmap_reader import BedMmapReader

    with BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants) as reader:
        packed = np.array(reader.read_all_packed(), copy=True)

    mean_np, scale_np = compute_mean_scale(packed, n_samples=n_samples, count_a1=True)
    mean_f32 = mean_np.astype(np.float32)
    scale_f32 = scale_np.astype(np.float32)

    packed_dev = cp.asarray(packed)
    mean_dev = cp.asarray(mean_f32)
    scale_dev = cp.asarray(scale_f32)

    matrix = BitpackedDeviceMatrix(
        packed=packed_dev,
        n_samples=n_samples,
        mean=mean_dev,
        std=scale_dev,
        count_a1=True,
    )

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal(n_variants).astype(np.float32)
    y_np = rng.standard_normal(n_samples).astype(np.float32)

    # matvec: G @ x ==> shape (n_samples,)
    out_mv = cp.asnumpy(matrix.matvec(cp.asarray(x_np)))
    ref_mv = cpu_gemv_nt(packed, n_samples, x_np, mean_f32, scale=scale_f32, count_a1=True)
    np.testing.assert_allclose(out_mv, ref_mv, rtol=5e-3, atol=5e-3)

    # rmatvec: G.T @ y ==> shape (n_variants,)
    out_rmv = cp.asnumpy(matrix.rmatvec(cp.asarray(y_np)))
    ref_rmv = cpu_gemv_tn(packed, n_samples, y_np, mean_f32, scale=scale_f32, count_a1=True)
    np.testing.assert_allclose(out_rmv, ref_rmv, rtol=5e-3, atol=5e-3)

    # gram block over a subset of variants.
    idx = np.arange(min(32, n_variants), dtype=np.int64)
    gram_dev = matrix.gram_block(idx)
    gram_host = cp.asnumpy(gram_dev)
    sub_packed = packed[idx]
    ref_gram = cpu_gemm_gram(
        sub_packed,
        n_samples,
        mean_f32[idx],
        scale=scale_f32[idx],
        count_a1=True,
    )
    np.testing.assert_allclose(gram_host, ref_gram, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    sys.exit(main())

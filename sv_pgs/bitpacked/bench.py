"""GPU benchmark harness for the bitpacked kernel pipeline.

Run as::

    python -m sv_pgs.bitpacked.bench [--output report.json] [--quick]

Reports GB/s for the three GEMV/screen kernels and TFLOPS for the gram kernel
at three representative scales (small / medium / large) chosen relative to the
detected HBM budget. GPU-only by design: CPU-only environments exit with a
clear message and a non-zero status.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any


def _require_gpu() -> Any:
    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001 - any failure means no GPU usable
        print(
            f"bench: cupy unavailable ({type(exc).__name__}: {exc}); "
            "this benchmark requires a CUDA GPU.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        cp.cuda.Device(0).compute_capability  # noqa: B018 - probe
    except Exception as exc:  # noqa: BLE001
        print(
            f"bench: no CUDA device available ({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
        sys.exit(2)
    return cp


@dataclass
class BenchRow:
    op: str
    n_samples: int
    n_variants: int
    bytes_gb: float
    time_ms: float
    gbps: float | None
    tflops: float | None


def _bytes_per_variant(n_samples: int) -> int:
    return (int(n_samples) + 3) // 4


def _alloc_synth(cp: Any, n_samples: int, n_variants: int, seed: int = 0xC0FFEE):
    """Allocate a random uint8-packed (n_variants, bpv) matrix and per-variant stats."""
    bpv = _bytes_per_variant(n_samples)
    rng = cp.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(n_variants, bpv), dtype=cp.uint8)
    # Ensure non-trivial mean/std so the kernels do real work.
    mean = cp.full((n_variants,), 1.0, dtype=cp.float32)
    std = cp.full((n_variants,), 0.7071, dtype=cp.float32)
    return packed, mean, std


def _bench_op(fn, n_warmup: int = 1, n_iter: int = 3) -> float:
    """Run ``fn`` ``n_iter`` times after warmup; return min wall-time in ms."""
    import cupy as cp

    for _ in range(n_warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    best = float("inf")
    for _ in range(n_iter):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        if elapsed < best:
            best = elapsed
    return best


def _bench_scale(cp: Any, n_samples: int, n_variants: int) -> list[BenchRow]:
    from sv_pgs.bitpacked.gemv_nt import gemv_nt
    from sv_pgs.bitpacked.gemv_tn import gemv_tn
    from sv_pgs.bitpacked.gemm_gram import gemm_gram
    from sv_pgs.bitpacked.screening import screen

    packed, mean, std = _alloc_synth(cp, n_samples, n_variants)
    bytes_gb = float(packed.nbytes) / 1e9
    rows: list[BenchRow] = []

    # ---- gemv_nt: y[n_samples] += Z @ x[n_variants]
    x = cp.ones((n_variants,), dtype=cp.float32)
    y = cp.zeros((n_samples,), dtype=cp.float32)
    t_ms = _bench_op(lambda: gemv_nt(packed, n_samples, x, mean, std, y))
    gbps = (packed.nbytes) / (t_ms * 1e-3) / 1e9
    rows.append(BenchRow("gemv_nt", n_samples, n_variants, bytes_gb, t_ms, gbps, None))

    # ---- gemv_tn: g[n_variants] += Z.T @ y[n_samples]
    yv = cp.ones((n_samples,), dtype=cp.float32)
    g = cp.zeros((n_variants,), dtype=cp.float32)
    t_ms = _bench_op(lambda: gemv_tn(packed, n_samples, yv, mean, std, g))
    gbps = (packed.nbytes) / (t_ms * 1e-3) / 1e9
    rows.append(BenchRow("gemv_tn", n_samples, n_variants, bytes_gb, t_ms, gbps, None))

    # ---- gemm_gram: B[n_variants, n_variants] += Z.T @ Z. FLOPS = 2 * n_samples * n_variants * (n_variants + 1) / 2.
    out = cp.zeros((n_variants, n_variants), dtype=cp.float32)
    t_ms = _bench_op(lambda: gemm_gram(packed, n_samples, mean, std, out))
    flops = 2.0 * float(n_samples) * float(n_variants) * float(n_variants + 1) / 2.0
    tflops = flops / (t_ms * 1e-3) / 1e12
    rows.append(BenchRow("gemm_gram", n_samples, n_variants, bytes_gb, t_ms, None, tflops))

    # ---- screening (count+sum+sumsq+rhs).
    rhs = cp.ones((n_samples,), dtype=cp.float32)
    out_count = cp.zeros((n_variants,), dtype=cp.int32)
    out_sum = cp.zeros((n_variants,), dtype=cp.float64)
    out_sumsq = cp.zeros((n_variants,), dtype=cp.float64)
    out_drhs = cp.zeros((n_variants,), dtype=cp.float64)
    out_orhs = cp.zeros((n_variants,), dtype=cp.float64)
    t_ms = _bench_op(
        lambda: screen(
            packed,
            n_samples,
            out_count,
            out_sum,
            out_sumsq,
            rhs=rhs,
            out_dosage_rhs=out_drhs,
            out_observed_rhs=out_orhs,
        )
    )
    gbps = (packed.nbytes) / (t_ms * 1e-3) / 1e9
    rows.append(BenchRow("screen", n_samples, n_variants, bytes_gb, t_ms, gbps, None))

    # Release intermediates between scales.
    del packed, mean, std, x, y, yv, g, out, rhs
    del out_count, out_sum, out_sumsq, out_drhs, out_orhs
    cp.get_default_memory_pool().free_all_blocks()
    return rows


def _pick_scales(total_hbm_gb: float, quick: bool) -> list[tuple[int, int]]:
    """Choose (n_samples, n_variants) tuples sized to the HBM budget.

    Each scale must fit ``packed`` plus ``out`` (gram, float32 n_variants^2)
    plus a 25% safety margin into HBM. We cap n_variants at a level where the
    gram output is the dominant footprint (gemm_gram's n_variants^2 fp32
    accumulator dominates for n_variants >> 1024).
    """
    # Budget the gram OUTPUT to ~30% of HBM (it's n_variants^2 * 4 bytes).
    # Packed budget is ~10% of HBM.
    budget_out_gb = max(0.2, total_hbm_gb * 0.30)
    max_n_variants_out = int((budget_out_gb * 1e9 / 4.0) ** 0.5)
    # Large scale: bounded by both the gram output and a fixed cohort size.
    large_v = min(8192, max_n_variants_out)
    medium_v = max(1024, large_v // 2)
    small_v = 512
    # Use a fixed sample count typical of cohort scale.
    n_samples = 97000
    scales: list[tuple[int, int]]
    if quick:
        scales = [(n_samples, small_v), (n_samples, medium_v)]
    else:
        scales = [(n_samples, small_v), (n_samples, medium_v), (n_samples, large_v)]
    return scales


def _print_table(rows: list[BenchRow]) -> None:
    print()
    print(
        "| Op        | n_samples | n_variants | bytes_GB | time_ms | GB/s   | TFLOPS |"
    )
    print(
        "|-----------|-----------|------------|----------|---------|--------|--------|"
    )
    for row in rows:
        gbps = f"{row.gbps:.1f}" if row.gbps is not None else "-"
        tflops = f"{row.tflops:.2f}" if row.tflops is not None else "-"
        print(
            f"| {row.op:<9} | {row.n_samples:>9} | {row.n_variants:>10} | "
            f"{row.bytes_gb:>8.3f} | {row.time_ms:>7.2f} | {gbps:>6} | {tflops:>6} |"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sv_pgs.bitpacked.bench")
    parser.add_argument("--output", type=str, default=None, help="JSON report path")
    parser.add_argument("--quick", action="store_true", help="Run 2 scales for fast CI")
    args = parser.parse_args(argv)

    cp = _require_gpu()
    from sv_pgs.bitpacked.launch import gpu_arch, gpu_arch_summary

    arch = gpu_arch()
    summary = gpu_arch_summary()
    device = cp.cuda.Device(arch.device_id)
    free_bytes, total_bytes = device.mem_info
    total_gb = total_bytes / 1e9
    free_gb = free_bytes / 1e9

    print(f"=== sv-pgs bitpacked benchmark on {summary} ===")
    print(f"HBM total: {total_gb:.1f} GB, free at start: {free_gb:.1f} GB")

    scales = _pick_scales(total_gb, quick=args.quick)
    print(f"running {len(scales)} scale(s): {scales}")

    all_rows: list[BenchRow] = []
    for n_samples, n_variants in scales:
        print(f"  scale: n_samples={n_samples}, n_variants={n_variants}")
        try:
            rows = _bench_scale(cp, n_samples, n_variants)
        except Exception as exc:  # noqa: BLE001
            print(f"    [FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        all_rows.extend(rows)

    _print_table(all_rows)

    report = {
        "arch": {
            "name": arch.name,
            "sm": arch.sm,
            "family": arch.family,
            "compute_capability": list(arch.compute_capability),
            "hbm_total_gb": total_gb,
            "hbm_free_gb_at_start": free_gb,
            "summary": summary,
        },
        "rows": [asdict(r) for r in all_rows],
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

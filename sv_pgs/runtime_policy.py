from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from sv_pgs.config import ModelConfig
from sv_pgs.genotype import RawGenotypeMatrix, _gpu_materialization_budget_bytes, _try_import_cupy

# Algorithmic limits — not GPU-memory-dependent.
# The exact solver limit caps dense Cholesky factorizations on GPU to avoid
# excessive O(p^3) cost. The preconditioner rank bounds the Nyström approximation.
GPU_PRECONDITIONER_RANK_FLOOR = 128
GPU_PRECONDITIONER_RANK_CEILING = 512
GPU_PRECONDITIONER_RANK_FRACTION = 0.04
GPU_STOCHASTIC_EXACT_GRAM_WORK_TARGET = 12_000_000_000_000.0


@dataclass(frozen=True, slots=True)
class RuntimeTrainingPolicy:
    tuned_config: ModelConfig
    gpu_budget_bytes: int | None
    cacheable_dense_variants: int | None


@dataclass(frozen=True, slots=True)
class GpuComputeSanityCheck:
    matrix_size: int
    repetitions: int
    elapsed_ms: float
    device_count: int


def _require_cupy() -> Any:
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    return cupy


def _gpu_memory_info(cupy: Any) -> tuple[int | None, int | None]:
    try:
        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    except (AttributeError, OSError, RuntimeError):
        return None, None
    return int(free_bytes), int(total_bytes)


def ensure_gpu_compute_active(
    *,
    matrix_size: int = 512,
    repetitions: int = 3,
    cupy: Any | None = None,
) -> GpuComputeSanityCheck:
    """Run a small timed CuPy matmul and fail if CUDA execution is broken."""
    cp = _require_cupy() if cupy is None else cupy
    if matrix_size < 1:
        raise ValueError("matrix_size must be positive.")
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("CuPy CUDA runtime is unavailable.") from exc
    if device_count < 1:
        raise RuntimeError("CuPy reports no CUDA devices.")

    try:
        a = cp.ones((matrix_size, matrix_size), dtype=cp.float32)
        b = cp.eye(matrix_size, dtype=cp.float32)
        cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        result = None
        for _ in range(int(repetitions)):
            result = cp.matmul(a, b)
        end.record()
        end.synchronize()
        elapsed_ms = float(cp.cuda.get_elapsed_time(start, end))
        if result is None:
            raise RuntimeError("CuPy matmul did not produce a result.")
        check_value = float(result[0, 0].get())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("CuPy GPU compute sanity check failed.") from exc

    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
        raise RuntimeError(f"CuPy GPU event timing is invalid: elapsed_ms={elapsed_ms!r}.")
    if not math.isclose(check_value, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError(f"CuPy matmul sanity check returned {check_value!r}; expected 1.0.")

    return GpuComputeSanityCheck(
        matrix_size=int(matrix_size),
        repetitions=int(repetitions),
        elapsed_ms=elapsed_ms,
        device_count=device_count,
    )


def _should_probe_cupy_compute(cupy: Any) -> bool:
    return (
        getattr(cupy, "__name__", None) == "cupy"
        and hasattr(cupy, "matmul")
        and hasattr(getattr(cupy, "cuda", None), "Event")
    )


def _recommended_gpu_preconditioner_rank(cacheable_dense_variants: int) -> int:
    if cacheable_dense_variants < 1:
        return GPU_PRECONDITIONER_RANK_FLOOR
    budget_scaled_rank = int(round(float(cacheable_dense_variants) * GPU_PRECONDITIONER_RANK_FRACTION))
    return max(
        GPU_PRECONDITIONER_RANK_FLOOR,
        min(
            GPU_PRECONDITIONER_RANK_CEILING,
            int(cacheable_dense_variants),
            budget_scaled_rank,
        ),
    )


def _device_scaled_exact_gram_work_target(total_gpu_bytes: int | None) -> float:
    """Scale the per-iteration exact-Gram FLOPs target with device capability.

    The baseline target is tuned for T4-class GPUs (~16 GB, ~8 TFLOPs fp32).
    On larger devices, fp32 throughput scales roughly linearly with
    HBM/SRAM-driven occupancy, so use device memory as a cheap proxy for
    arithmetic capability without hardcoding a TFLOPs lookup table:

      - 16 GB (T4):  1.0x   = 12 TFLOPs target
      - 40 GB (A100-40): ~3x = 36 TFLOPs target
      - 80 GB (A100-80/H100): ~5x+ = 60+ TFLOPs target

    This keeps stochastic blocks in the exact-GPU regime on large devices
    instead of fragmenting into many small batches and undersaturating the
    SMs. Bytes are converted to fraction-of-T4-memory so the heuristic is
    device-agnostic.
    """
    baseline = float(GPU_STOCHASTIC_EXACT_GRAM_WORK_TARGET)
    if total_gpu_bytes is None or total_gpu_bytes <= 0:
        return baseline
    t4_reference_bytes = 16.0 * 1e9
    scale = max(1.0, float(total_gpu_bytes) / t4_reference_bytes)
    return baseline * scale


def _recommended_gpu_stochastic_batch_size(
    *,
    cacheable_dense_variants: int,
    sample_count: int,
    total_gpu_bytes: int | None = None,
) -> int:
    if cacheable_dense_variants < 1:
        return 256
    dense_budget_batch_size = int(cacheable_dense_variants * 0.85)
    scaled_work_target = _device_scaled_exact_gram_work_target(total_gpu_bytes)
    exact_gpu_work_batch_size = int(
        math.sqrt(
            scaled_work_target
            / max(float(sample_count), 1.0)
        )
    )
    # Pick the smaller of dense-budget vs exact-Gram work limit. Do NOT floor
    # at 256 unconditionally: when the GPU can only hold a handful of dense
    # variants, forcing 256 exceeds the measured budget and OOMs the block.
    # Apply the 256 floor only if budget permits.
    budget_capped = min(dense_budget_batch_size, exact_gpu_work_batch_size)
    if budget_capped >= 256:
        return budget_capped
    return max(budget_capped, 1)


def runtime_training_policy_for_fit(
    config: ModelConfig,
    genotype_matrix: RawGenotypeMatrix,
) -> RuntimeTrainingPolicy:
    cupy = _try_import_cupy()
    if cupy is None:
        return RuntimeTrainingPolicy(
            tuned_config=config,
            gpu_budget_bytes=None,
            cacheable_dense_variants=None,
        )
    sample_count = int(genotype_matrix.shape[0])
    if sample_count < 1:
        return RuntimeTrainingPolicy(
            tuned_config=config,
            gpu_budget_bytes=None,
            cacheable_dense_variants=None,
        )
    if _should_probe_cupy_compute(cupy):
        ensure_gpu_compute_active(cupy=cupy)
    gpu_budget_bytes = _gpu_materialization_budget_bytes(cupy)
    cacheable_dense_variants = max(int(gpu_budget_bytes // max(sample_count * 4, 1)), 1)
    tuned_exact_solver_limit = min(
        int(config.exact_solver_matrix_limit),
        max(int(cacheable_dense_variants * 0.9), 1),
    )
    if int(config.sample_space_preconditioner_rank) <= 0:
        tuned_preconditioner_rank = 0
    else:
        max_gpu_preconditioner_rank = max(1, int(cacheable_dense_variants))
        recommended_preconditioner_rank = _recommended_gpu_preconditioner_rank(cacheable_dense_variants)
        tuned_preconditioner_rank = min(
            max(int(config.sample_space_preconditioner_rank), recommended_preconditioner_rank),
            max_gpu_preconditioner_rank,
        )
    # Use as much dense GPU budget as the exact variant-space Gram build can
    # use efficiently. On very large cohorts, this keeps stochastic blocks in
    # the exact-GPU solve regime instead of drifting into slower sample-space CG.
    # Do not re-floor at 256 here — the helper already caps the floor by the
    # dense budget so tight-GPU runs can return a smaller (but feasible) value.
    _, total_gpu_bytes = _gpu_memory_info(cupy)
    tuned_stochastic_batch_size = _recommended_gpu_stochastic_batch_size(
        cacheable_dense_variants=cacheable_dense_variants,
        sample_count=sample_count,
        total_gpu_bytes=total_gpu_bytes,
    )
    tuned_config = replace(
        config,
        exact_solver_matrix_limit=tuned_exact_solver_limit,
        sample_space_preconditioner_rank=tuned_preconditioner_rank,
        stochastic_variant_batch_size=max(tuned_stochastic_batch_size, 1),
    )
    return RuntimeTrainingPolicy(
        tuned_config=tuned_config,
        gpu_budget_bytes=gpu_budget_bytes,
        cacheable_dense_variants=cacheable_dense_variants,
    )


def runtime_training_policy_summary(policy: RuntimeTrainingPolicy, original_config: ModelConfig) -> str | None:
    if policy.gpu_budget_bytes is None or policy.cacheable_dense_variants is None:
        return None
    tuned_config = policy.tuned_config
    if tuned_config == original_config:
        return (
            "GPU runtime profile active: "
            + f"gpu_budget={policy.gpu_budget_bytes / 1e9:.1f} GB "
            + f"cacheable_dense_variants~{policy.cacheable_dense_variants} "
            + "gpu_profile=budget-driven "
            + "(user config already fits GPU profile)"
        )
    return (
        "GPU runtime profile active: "
        + f"gpu_budget={policy.gpu_budget_bytes / 1e9:.1f} GB "
        + f"cacheable_dense_variants~{policy.cacheable_dense_variants} "
        + "gpu_profile=budget-driven "
        + f"exact_solver_matrix_limit={original_config.exact_solver_matrix_limit}->{tuned_config.exact_solver_matrix_limit} "
        + f"sample_space_preconditioner_rank={original_config.sample_space_preconditioner_rank}->{tuned_config.sample_space_preconditioner_rank} "
        + f"stochastic_variant_batch_size={original_config.stochastic_variant_batch_size}->{tuned_config.stochastic_variant_batch_size}"
    )

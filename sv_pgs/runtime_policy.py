from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import math
from dataclasses import dataclass, replace
import logging
import time
from typing import Any, Literal

from sv_pgs.config import ModelConfig
from sv_pgs.genotype import RawGenotypeMatrix, _gpu_materialization_budget_bytes, _try_import_cupy

# Algorithmic limits — not GPU-memory-dependent.
# The exact solver limit caps dense Cholesky factorizations on GPU to avoid
# excessive O(p^3) cost. The preconditioner rank bounds the Nyström approximation.
GPU_PRECONDITIONER_RANK_FLOOR = 128
GPU_PRECONDITIONER_RANK_CEILING = 512
GPU_PRECONDITIONER_RANK_FRACTION = 0.04
GPU_STOCHASTIC_EXACT_GRAM_WORK_TARGET = 12_000_000_000_000.0
GPU_ALLOCATOR_DEFAULT_POOL_FRACTION = 0.72
GPU_PHASE_POOL_GROWTH_RELEASE_THRESHOLD_BYTES = 512 * 1024 * 1024

GpuAllocatorStrategy = Literal["bounded_pool", "no_pool", "managed"]

_LOGGER = logging.getLogger(__name__)
_GPU_ALLOCATOR_POOL: Any | None = None
_GPU_STAGING_BUFFERS: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class RuntimeTrainingPolicy:
    tuned_config: ModelConfig
    gpu_budget_bytes: int | None
    cacheable_dense_variants: int | None


@dataclass(frozen=True, slots=True)
class GpuAllocatorPolicy:
    strategy: GpuAllocatorStrategy
    max_pool_bytes: int | None
    total_gpu_bytes: int | None


@dataclass(frozen=True, slots=True)
class GpuComputeSanityCheck:
    matrix_size: int
    repetitions: int
    elapsed_ms: float
    device_count: int


@dataclass(frozen=True, slots=True)
class _GpuMemorySnapshot:
    pool_used_bytes: int
    pool_total_bytes: int
    device_free_bytes: int | None
    device_total_bytes: int | None


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


def _pool_bytes(pool: Any, method_name: str) -> int:
    method = getattr(pool, method_name, None)
    if method is None:
        return 0
    try:
        return int(method())
    except (OSError, RuntimeError, TypeError, ValueError):
        return 0


def _gpu_memory_snapshot(cupy: Any) -> _GpuMemorySnapshot:
    pool = cupy.get_default_memory_pool()
    free_bytes, total_bytes = _gpu_memory_info(cupy)
    return _GpuMemorySnapshot(
        pool_used_bytes=_pool_bytes(pool, "used_bytes"),
        pool_total_bytes=_pool_bytes(pool, "total_bytes"),
        device_free_bytes=free_bytes,
        device_total_bytes=total_bytes,
    )


def _format_gpu_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1e9:.2f} GB"


def _log_gpu_policy(message: str) -> None:
    try:
        from sv_pgs.progress import log
    except (ImportError, OSError, RuntimeError):
        _LOGGER.info(message)
    else:
        log(message)


def _pool_limit(pool: Any) -> int | None:
    get_limit = getattr(pool, "get_limit", None)
    if get_limit is None:
        return None
    try:
        return int(get_limit())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _set_pool_limit(pool: Any, max_pool_bytes: int | None) -> None:
    set_limit = getattr(pool, "set_limit", None)
    if set_limit is None:
        return
    size = None if max_pool_bytes is None else int(max_pool_bytes)
    try:
        set_limit(size=size)
    except TypeError:
        if size is None:
            return
        set_limit(size)


def _free_pool_blocks(cupy: Any) -> None:
    for pool_getter_name in ("get_default_memory_pool", "get_default_pinned_memory_pool"):
        pool_getter = getattr(cupy, pool_getter_name, None)
        if pool_getter is None:
            continue
        try:
            pool = pool_getter()
            pool.free_all_blocks()
        except (AttributeError, OSError, RuntimeError):
            continue


def _bounded_pool_bytes(
    *,
    total_gpu_bytes: int | None,
    pool_fraction: float,
    max_pool_bytes: int | None,
) -> int:
    if max_pool_bytes is not None:
        limit = int(max_pool_bytes)
    else:
        if total_gpu_bytes is None or total_gpu_bytes <= 0:
            raise RuntimeError("GPU total memory is unavailable; pass max_pool_bytes explicitly.")
        if not 0.0 < float(pool_fraction) <= 1.0:
            raise ValueError("pool_fraction must be in the interval (0, 1].")
        limit = int(total_gpu_bytes * float(pool_fraction))
    if limit <= 0:
        raise ValueError("GPU memory pool limit must be positive.")
    return limit


def configure_gpu_allocator(
    strategy: GpuAllocatorStrategy = "bounded_pool",
    *,
    pool_fraction: float = GPU_ALLOCATOR_DEFAULT_POOL_FRACTION,
    max_pool_bytes: int | None = None,
) -> GpuAllocatorPolicy:
    """Configure CuPy allocation before memory-heavy GPU phases."""
    global _GPU_ALLOCATOR_POOL

    cupy = _require_cupy()
    _, total_gpu_bytes = _gpu_memory_info(cupy)
    cuda = getattr(cupy, "cuda", None)
    set_allocator = getattr(cuda, "set_allocator", None)
    if set_allocator is None:
        raise RuntimeError("CuPy CUDA allocator API is unavailable.")

    if strategy == "bounded_pool":
        pool = cupy.get_default_memory_pool()
        limit = _bounded_pool_bytes(
            total_gpu_bytes=total_gpu_bytes,
            pool_fraction=pool_fraction,
            max_pool_bytes=max_pool_bytes,
        )
        _set_pool_limit(pool, limit)
        set_allocator(pool.malloc)
        _GPU_ALLOCATOR_POOL = pool
        _free_pool_blocks(cupy)
        _log_gpu_policy(f"CuPy allocator configured: strategy=bounded_pool pool_limit={_format_gpu_bytes(limit)}")
        return GpuAllocatorPolicy(strategy=strategy, max_pool_bytes=limit, total_gpu_bytes=total_gpu_bytes)

    if strategy == "no_pool":
        _free_pool_blocks(cupy)
        set_allocator(None)
        _GPU_ALLOCATOR_POOL = None
        _log_gpu_policy("CuPy allocator configured: strategy=no_pool")
        return GpuAllocatorPolicy(strategy=strategy, max_pool_bytes=None, total_gpu_bytes=total_gpu_bytes)

    if strategy == "managed":
        memory_pool = getattr(cuda, "MemoryPool", None)
        malloc_managed = getattr(cuda, "malloc_managed", None)
        if memory_pool is None or malloc_managed is None:
            raise RuntimeError("CuPy managed memory allocator API is unavailable.")
        pool = memory_pool(malloc_managed)
        if max_pool_bytes is not None:
            _set_pool_limit(pool, int(max_pool_bytes))
        set_allocator(pool.malloc)
        _GPU_ALLOCATOR_POOL = pool
        _log_gpu_policy(
            "CuPy allocator configured: strategy=managed "
            + f"pool_limit={_format_gpu_bytes(max_pool_bytes)}"
        )
        return GpuAllocatorPolicy(
            strategy=strategy,
            max_pool_bytes=None if max_pool_bytes is None else int(max_pool_bytes),
            total_gpu_bytes=total_gpu_bytes,
        )

    raise ValueError(f"Unknown CuPy allocator strategy: {strategy!r}")


def preallocate_staging(shape: Sequence[int], dtype: Any, name: str) -> Any:
    """Return a named reusable CuPy staging buffer."""
    cupy = _require_cupy()
    if not name:
        raise ValueError("staging buffer name must be non-empty.")
    resolved_shape = tuple(int(dimension) for dimension in shape)
    if any(dimension < 0 for dimension in resolved_shape):
        raise ValueError("staging buffer shape dimensions must be non-negative.")
    resolved_dtype = cupy.dtype(dtype)
    key = str(name)
    buffer = _GPU_STAGING_BUFFERS.get(key)
    if buffer is not None:
        if tuple(buffer.shape) != resolved_shape or cupy.dtype(buffer.dtype) != resolved_dtype:
            raise ValueError(
                f"staging buffer {key!r} already exists with "
                + f"shape={tuple(buffer.shape)!r} dtype={buffer.dtype!r}; "
                + f"requested shape={resolved_shape!r} dtype={resolved_dtype!r}"
            )
        return buffer
    buffer = cupy.empty(resolved_shape, dtype=resolved_dtype)
    _GPU_STAGING_BUFFERS[key] = buffer
    return buffer


@contextmanager
def bounded_gpu_phase(name: str, max_pool_bytes: int | None = None) -> Iterator[None]:
    """Bound CuPy pool growth and release cached blocks after leaky phases."""
    cupy = _require_cupy()
    pool = cupy.get_default_memory_pool()
    previous_limit = _pool_limit(pool)
    start = _gpu_memory_snapshot(cupy)
    start_time = time.monotonic()

    if max_pool_bytes is not None:
        _set_pool_limit(pool, int(max_pool_bytes))

    try:
        yield
    finally:
        end = _gpu_memory_snapshot(cupy)
        elapsed_seconds = time.monotonic() - start_time
        pool_total_delta = end.pool_total_bytes - start.pool_total_bytes
        pool_used_delta = end.pool_used_bytes - start.pool_used_bytes
        if end.device_free_bytes is None or start.device_free_bytes is None:
            device_free_delta = None
        else:
            device_free_delta = end.device_free_bytes - start.device_free_bytes

        if pool_total_delta > GPU_PHASE_POOL_GROWTH_RELEASE_THRESHOLD_BYTES:
            _free_pool_blocks(cupy)
            release_note = " cached_blocks_released=true"
        else:
            release_note = " cached_blocks_released=false"

        if max_pool_bytes is not None and previous_limit is not None:
            _set_pool_limit(pool, previous_limit)

        _log_gpu_policy(
            f"GPU phase {name}: elapsed={elapsed_seconds:.1f}s "
            + f"pool_used_delta={_format_gpu_bytes(pool_used_delta)} "
            + f"pool_total_delta={_format_gpu_bytes(pool_total_delta)} "
            + f"device_free_delta={_format_gpu_bytes(device_free_delta)}"
            + release_note
        )


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


def _recommended_gpu_stochastic_batch_size(
    *,
    cacheable_dense_variants: int,
    sample_count: int,
) -> int:
    if cacheable_dense_variants < 1:
        return 256
    dense_budget_batch_size = int(cacheable_dense_variants * 0.85)
    exact_gpu_work_batch_size = int(
        math.sqrt(
            GPU_STOCHASTIC_EXACT_GRAM_WORK_TARGET
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
    tuned_stochastic_batch_size = _recommended_gpu_stochastic_batch_size(
        cacheable_dense_variants=cacheable_dense_variants,
        sample_count=sample_count,
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

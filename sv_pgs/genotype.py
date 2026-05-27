from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time as _time_module
from typing import Any, Iterator, Protocol, Sequence, TypeGuard, cast

import sv_pgs._jax as _jax_side_effects  # side-effect: configures JAX/XLA env
del _jax_side_effects
import jax
from jax import core as jax_core
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
import numpy as np
from sv_pgs._jax import gpu_compute_jax_dtype, gpu_compute_numpy_dtype, jax_dense_linear_algebra_preferred
from sv_pgs._typing import JaxArray, NDArray
from sv_pgs.plink import PLINK_MISSING_INT8, open_bed
from sv_pgs.progress import log, mem

DEFAULT_GENOTYPE_BATCH_SIZE = 1024  # fallback when sample count is unknown

# ---------------------------------------------------------------------------
# Runtime auto-tuning of memory/IO sizing.
#
# Historically these were hand-tuned constants (500 MB BED batches, prefetch
# depth 3). That works well on the "blessed" T4/16-core box but starves a
# 2-core/V100 box (prefetch eats the only available cores) and under-utilizes
# a 64-core/H100 box (batches finish faster than disk can refill). The
# helpers below detect available RAM/CPU/GPU at import time and pick values
# that scale with the actual hardware. The constants themselves are still
# set at import so existing callers that do
# ``from sv_pgs.genotype import BED_READER_TARGET_BATCH_BYTES`` keep working.
# ---------------------------------------------------------------------------

_AUTO_TUNE_HOST_RAM_FALLBACK_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
_AUTO_TUNE_BED_BATCH_HARD_CAP_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
_AUTO_TUNE_BED_BATCH_FLOOR_BYTES = 128 * 1024 * 1024  # 128 MB
_AUTO_TUNE_PREFETCH_DEPTH_HARD_CAP = 4  # I/O-bound BED reads (~3 s each) don't
# benefit from >4 concurrent slots — more just balloons inflight RAM. The hard
# cap is per-shard, so a 2-GPU box can still have 8 readers total.
_AUTO_TUNE_HOST_RAM_INFLIGHT_FRACTION = 0.40  # ≤ 40% of usable RAM (after fit
# reserve) may be tied up in prefetch buffers across all shards combined.
# Higher than the prior 0.25 because the reserve already accounts for fit working
# set; lower than 1.0 because cupy pool, Python heap, and OS overhead also need
# room. On a 30 GB box: usable≈20 GB → inflight cap≈8 GB.
_AUTO_TUNE_GPU_BATCH_FRACTION = 0.40  # one batch fits in 40% of GPU free
# Reserved RAM for the fit working set itself (model params, accumulators,
# Hessian-vector workspace, JAX/XLA scratch). Subtracted from MemAvailable
# BEFORE sizing the prefetch budget so the prefetch queue can never consume
# all of MemAvailable and OOM the trainer mid-epoch. 4 GB is the empirically
# observed AoU peak (T4/V100, 695k variants × 331k samples) plus headroom;
# the May 2026 30 GB-container OOM (rc=137 RSS=11 GB) traced to prefetch
# eating 8 GB while the fit needed ~3 GB working set + ~2 GB accumulators.
_AUTO_TUNE_FIT_WORKING_MEM_RESERVE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
# (was 4 GB — the May 2026 30 GB OOM at RSS=29 GB showed actual fit working
# set + cupy pool + sharded-matmul accumulators + Python heap routinely
# consumes 8-10 GB, not 4. The prefetch budget MUST exclude this or each
# shard's inflight queue cumulatively pushes total RSS past MemAvailable.)
_AUTO_TUNE_MIN_USABLE_PREFETCH_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB floor
# Cap a single batch at half of usable prefetch RAM so at least two batches
# can be queued (otherwise prefetch_depth collapses to 1 and the GPU stalls).
_AUTO_TUNE_BED_BATCH_USABLE_FRACTION = 0.5


def _parse_proc_meminfo() -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a {key: bytes} mapping.

    Returns an empty dict on any error (e.g. non-Linux, unreadable file).
    """
    result: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as meminfo:
            for line in meminfo:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                try:
                    value_kb = int(parts[1])
                except ValueError:
                    continue
                # /proc/meminfo reports kB (i.e. KiB) for memory rows.
                unit = parts[2].lower() if len(parts) >= 3 else "kb"
                if unit == "kb":
                    result[key] = value_kb * 1024
                else:
                    result[key] = value_kb
    except OSError:
        return {}
    return result


def _detect_available_host_ram_bytes() -> int:
    """Return available host RAM in bytes.

    Precedence:
        1. ``/proc/meminfo:MemAvailable`` (Linux 3.14+; authoritative)
        2. ``MemFree + Cached + SReclaimable`` from ``/proc/meminfo``
           (manual MemAvailable approximation for ancient kernels)
        3. ``psutil.virtual_memory().available`` if psutil is importable
        4. ``SC_AVPHYS_PAGES * SC_PAGE_SIZE`` (MemFree-equivalent; pessimistic)
        5. 4 GB hardcoded floor
    """
    meminfo = _parse_proc_meminfo()
    if "MemAvailable" in meminfo and meminfo["MemAvailable"] > 0:
        return int(meminfo["MemAvailable"])
    if meminfo:
        approx = (
            meminfo.get("MemFree", 0)
            + meminfo.get("Cached", 0)
            + meminfo.get("SReclaimable", 0)
        )
        if approx > 0:
            return int(approx)
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        try:
            available = int(psutil.virtual_memory().available)
            if available > 0:
                return available
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        if page_size > 0 and avail_pages > 0:
            return int(page_size) * int(avail_pages)
    except (AttributeError, ValueError, OSError):
        pass
    return _AUTO_TUNE_HOST_RAM_FALLBACK_BYTES


def _cupy_runtime_error_classes(cupy: Any) -> tuple[type[BaseException], ...]:
    runtime = getattr(getattr(cupy, "cuda", None), "runtime", None)
    cuda_error = getattr(runtime, "CUDARuntimeError", None)
    classes: list[type[BaseException]] = [
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ]
    if isinstance(cuda_error, type) and issubclass(cuda_error, BaseException) and cuda_error not in classes:
        classes.append(cuda_error)
    return tuple(classes)


def _cupy_runtime_usable(cupy: Any) -> bool:
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except _cupy_runtime_error_classes(cupy):
        return False


def _detect_gpu_free_bytes() -> int:
    """Best-effort GPU free-memory probe via CuPy. Returns 0 if no GPU."""
    try:
        import cupy  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError):
        return 0
    try:
        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except _cupy_runtime_error_classes(cupy):
        return 0


def _detect_cuda_device_count() -> int:
    """Best-effort CUDA device count via CuPy. Returns 0 if no GPU."""
    try:
        import cupy  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError):
        return 0
    try:
        return max(int(cupy.cuda.runtime.getDeviceCount()), 0)
    except _cupy_runtime_error_classes(cupy):
        return 0


def _cupy_runtime_diagnostic() -> str:
    try:
        import cupy  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError) as exc:
        return f"cupy_import_error={exc.__class__.__name__}: {exc}"
    parts = [f"cupy={getattr(cupy, '__version__', '<unknown>')}"]
    try:
        device_count = max(int(cupy.cuda.runtime.getDeviceCount()), 0)
    except _cupy_runtime_error_classes(cupy) as exc:
        return " ".join(parts + [f"cupy_runtime_error={exc.__class__.__name__}: {exc}"])
    parts.append(f"cupy_cuda_devices={device_count}")
    for device_id in range(device_count):
        try:
            with cupy.cuda.Device(device_id):
                free, total = cupy.cuda.runtime.memGetInfo()
            parts.append(
                f"device{device_id}={free / 1e9:.1f}GB_free/{total / 1e9:.1f}GB_total"
            )
        except _cupy_runtime_error_classes(cupy) as exc:
            parts.append(f"device{device_id}_error={exc.__class__.__name__}: {exc}")
    return " ".join(parts)


def _nvidia_driver_diagnostic() -> str:
    command = shutil.which("nvidia-smi")
    if command is None:
        driver_version = Path("/proc/driver/nvidia/version")
        if driver_version.exists():
            try:
                return "nvidia-smi=missing " + driver_version.read_text(encoding="utf-8").strip().replace("\n", " | ")
            except OSError as exc:
                return f"nvidia-smi=missing nvidia_proc_version_error={exc}"
        device_files = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
        return "nvidia-smi=missing /dev=" + (",".join(device_files) if device_files else "<none>")
    import subprocess

    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=index,name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi_error={exc}"
    if result.returncode != 0:
        stderr = result.stderr.strip().replace("\n", " | ")
        return f"nvidia-smi_rc={result.returncode} stderr={stderr}"
    lines = " | ".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    return "nvidia-smi=" + (lines if lines else "no_visible_gpus")


def compute_usable_prefetch_ram_bytes(host_ram_bytes: int | None = None) -> int:
    """Return RAM bytes available for the prefetch queue after reserving fit working set.

    The autotune reserves :data:`_AUTO_TUNE_FIT_WORKING_MEM_RESERVE_BYTES` (4 GB)
    for the fit itself (model params, gradient accumulators, JAX scratch,
    Hessian-vector products). The prefetch queue may only consume what is
    left, with a 1 GB floor so degenerate boxes still make forward progress.
    """
    if host_ram_bytes is None:
        host_ram_bytes = _detect_available_host_ram_bytes()
    usable = int(host_ram_bytes) - _AUTO_TUNE_FIT_WORKING_MEM_RESERVE_BYTES
    return max(usable, _AUTO_TUNE_MIN_USABLE_PREFETCH_BYTES)


def compute_bed_reader_target_batch_bytes(
    *,
    host_ram_bytes: int | None = None,
    gpu_free_bytes: int | None = None,
    prefetch_depth: int | None = None,
    n_shards: int | None = None,
) -> int:
    """Auto-tune per-batch byte budget for the BED reader.

    Reserves :data:`_AUTO_TUNE_FIT_WORKING_MEM_RESERVE_BYTES` for the fit
    working set first, then sizes a single batch as a small fraction of the
    remaining usable RAM divided across shards and prefetch slots.

    Final formula:
        usable = max(host_ram_free - fit_reserve_4GB, 1 GB)
        host_share = usable / (prefetch_depth * n_shards)
        cap_per_batch = usable * 0.5  # so >=2 slots always queueable
        min(host_share, cap_per_batch, gpu_free * 0.40, 4 GB hard cap)
    Clamped to >= 128 MB floor.
    """
    if host_ram_bytes is None:
        host_ram_bytes = _detect_available_host_ram_bytes()
    if gpu_free_bytes is None:
        gpu_free_bytes = _detect_gpu_free_bytes()
    if prefetch_depth is None or prefetch_depth < 1:
        prefetch_depth = max(1, os.cpu_count() or 1)
    if n_shards is None or n_shards < 1:
        n_shards = max(1, _detect_cuda_device_count() or 1)
    usable_prefetch = compute_usable_prefetch_ram_bytes(host_ram_bytes)
    # Total inflight = prefetch_depth * n_shards batches. Size each batch so
    # the SUM of inflight bytes stays within `usable_prefetch`.
    total_slots = max(1, int(prefetch_depth) * int(n_shards))
    host_share = usable_prefetch // total_slots
    # Per-batch ceiling so a single batch can't monopolize prefetch RAM.
    per_batch_ceiling = int(usable_prefetch * _AUTO_TUNE_BED_BATCH_USABLE_FRACTION)
    candidates: list[int] = []
    if host_share > 0:
        candidates.append(host_share)
    if per_batch_ceiling > 0:
        candidates.append(per_batch_ceiling)
    if gpu_free_bytes > 0:
        candidates.append(int(gpu_free_bytes * _AUTO_TUNE_GPU_BATCH_FRACTION))
    candidates.append(_AUTO_TUNE_BED_BATCH_HARD_CAP_BYTES)
    chosen = min(candidates) if candidates else 500_000_000
    return max(chosen, _AUTO_TUNE_BED_BATCH_FLOOR_BYTES)


def compute_plink_int8_max_prefetch_depth(
    *,
    cpu_count: int | None = None,
    host_ram_bytes: int | None = None,
    target_batch_bytes: int | None = None,
    n_shards: int | None = None,
) -> int:
    """Auto-tune the int8 reader prefetch queue depth (per shard).

    Scales with cpu_count, capped by how many in-flight batches fit within
    the usable prefetch RAM budget (i.e. MemAvailable minus the fit working
    set reserve) ACROSS all GPU shards. When ``n_shards >= 2`` (multi-GPU
    sharded streaming), each shard runs its own prefetch executor — so the
    TOTAL RAM cost is ``depth × batch_bytes × n_shards``. This function
    returns the per-shard depth that respects that total. ``n_shards=1``
    reproduces the historical single-GPU sizing exactly.

    Always returns at least 1.
    """
    if cpu_count is None:
        cpu_count = max(1, os.cpu_count() or 1)
    if host_ram_bytes is None:
        host_ram_bytes = _detect_available_host_ram_bytes()
    if n_shards is None or n_shards < 1:
        n_shards = max(1, _detect_cuda_device_count() or 1)
    if target_batch_bytes is None:
        target_batch_bytes = compute_bed_reader_target_batch_bytes(
            host_ram_bytes=host_ram_bytes,
            prefetch_depth=cpu_count,
            n_shards=n_shards,
        )
    usable_prefetch = compute_usable_prefetch_ram_bytes(host_ram_bytes)
    # Total inflight across all shards must fit in usable RAM.
    total_capacity_batches = max(1, usable_prefetch // max(target_batch_bytes, 1))
    per_shard_ram_cap = max(1, total_capacity_batches // max(int(n_shards), 1))
    depth = min(int(cpu_count), int(per_shard_ram_cap), _AUTO_TUNE_PREFETCH_DEPTH_HARD_CAP)
    return max(1, depth)


def _snapshot_autotune_state() -> dict[str, int]:
    """Return current auto-tune detection snapshot for diagnostic logging."""
    host_ram = _detect_available_host_ram_bytes()
    gpu_free = _detect_gpu_free_bytes()
    cuda_device_count = _detect_cuda_device_count()
    cpu_count = max(1, os.cpu_count() or 1)
    n_shards = max(1, int(cuda_device_count) or 1)
    usable_prefetch = compute_usable_prefetch_ram_bytes(host_ram)
    bed_bytes = compute_bed_reader_target_batch_bytes(
        host_ram_bytes=host_ram,
        gpu_free_bytes=gpu_free,
        prefetch_depth=cpu_count,
        n_shards=n_shards,
    )
    depth = compute_plink_int8_max_prefetch_depth(
        cpu_count=cpu_count,
        host_ram_bytes=host_ram,
        target_batch_bytes=bed_bytes,
        n_shards=n_shards,
    )
    return {
        "cpu_count": cpu_count,
        "cuda_device_count": int(cuda_device_count),
        "host_ram_available_bytes": int(host_ram),
        "fit_working_mem_reserve_bytes": int(_AUTO_TUNE_FIT_WORKING_MEM_RESERVE_BYTES),
        "usable_prefetch_ram_bytes": int(usable_prefetch),
        "gpu_free_bytes": int(gpu_free),
        "bed_reader_target_batch_bytes": int(bed_bytes),
        "n_shards": int(n_shards),
        "plink_int8_max_prefetch_depth": int(depth),
        "plink_int8_max_prefetch_depth_total": int(depth) * int(n_shards),
        "per_worker_threads": max(1, int(cpu_count) // max(int(depth), 1)),
    }


# Auto-tuned at import time.
BED_READER_TARGET_BATCH_BYTES = compute_bed_reader_target_batch_bytes()
MIN_BED_READER_BATCH_SIZE = 32  # always read at least this many variants
STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 1_024_000_000
LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 4_096_000_000
GPU_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 512_000_000
GPU_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 4_096_000_000
GPU_STANDARDIZED_PREFETCH_TARGET_BYTES = 4_096_000_000
GPU_STANDARDIZED_DYNAMIC_FREE_FRACTION = 0.20
GPU_INT8_STANDARDIZED_DYNAMIC_FREE_FRACTION = 0.35
GPU_STANDARDIZED_DYNAMIC_RESERVE_BYTES = 512_000_000
PLINK_INT8_TARGET_BATCH_BYTES = 1_024_000_000
# Auto-tuned: scales with cpu_count and available host RAM.
PLINK_INT8_MAX_PREFETCH_DEPTH = compute_plink_int8_max_prefetch_depth(
    target_batch_bytes=BED_READER_TARGET_BATCH_BYTES,
)
PLINK_BED_READER_NUM_THREADS = max(1, os.cpu_count() or 1)
# Cap on rows promoted per int8 matmul staging chunk. With ~695k AoU variants
# this slab is ``chunk_rows x n_cols x 4 bytes`` (fp32 promotion); 4096 rows is
# ~11 GB on AoU and is intentionally too big to default to — the adaptive
# planner picks the chunk that actually fits in free VRAM (capped here).
GPU_INT8_MATMUL_STAGING_ROWS = 4096
# Hard upper bound on the adaptive chunk size — protects against pathological
# free-memory readings from blowing the staging slab past a sane size.
GPU_INT8_MATMUL_STAGING_ROWS_MAX = 8192
# Adaptive staging may consume up to this fraction of currently-free VRAM. The
# rest is reserved for the int8 cache, sample/variant result vectors, and the
# 1.5 GB TR-Newton/HVP safety margin.
GPU_INT8_MATMUL_STAGING_FREE_FRACTION = 0.40
# Absolute ceiling for the adaptive staging slab. Keeps the slab bounded even
# on very-high-memory GPUs (H100 80 GB) where 40% of free could be excessive.
GPU_INT8_MATMUL_STAGING_MAX_BYTES = 6_000_000_000
GPU_FP16_RESIDENT_CACHE_ENABLED = True

# If the reduced genotype matrix (after tie-group dedup) is smaller than 4 GB,
# cache it in RAM.  This avoids re-reading from disk on every EM iteration
# (typically 10-30 iterations), giving a huge speedup.
MATERIALIZE_THRESHOLD_BYTES = 4_000_000_000  # 4 GB
HYBRID_SPARSE_SUPPORT_THRESHOLD = 4_096
HYBRID_SPARSE_MIN_VARIANT_COUNT = 64
REDUCED_INT8_CACHE_FREE_SPACE_RESERVE_BYTES = 64 * 1024 * 1024
INT8_ONE_SHOT_GPU_BUDGET_FRACTION = 0.90
ROW_SUBSET_ONE_SHOT_MAX_SAMPLE_RATIO = 8.0

# Throttle for the per-batch "int8 batch:" log line. Each entry maps a
# (matrix_id, cache_key, batch_size) tuple to the monotonic time it last
# emitted; we re-log at most once per _INT8_BATCH_LOG_MIN_INTERVAL_SEC per key.
_INT8_BATCH_LOG_MIN_INTERVAL_SEC = 30.0
_int8_batch_log_last: dict[tuple[int, Any, int], float] = {}
_int8_batch_log_lock = threading.Lock()


def _log_int8_batch_throttled(
    *,
    matrix_id: int,
    cache_key: Any,
    batch_size: int,
    sample_count: int,
    batch_mb: float,
    n_batches: int,
) -> None:
    """Emit the per-batch int8 log line at most once per key per interval."""
    key = (matrix_id, cache_key, batch_size)
    now = _time_module.monotonic()
    with _int8_batch_log_lock:
        last = _int8_batch_log_last.get(key)
        if last is not None and (now - last) < _INT8_BATCH_LOG_MIN_INTERVAL_SEC:
            return
        _int8_batch_log_last[key] = now
    log(
        f"    int8 batch: {batch_size} variants x {sample_count} samples = "
        f"{batch_mb:.0f} MB/batch, {n_batches} batches  mem={mem()}"
    )


def _madvise_willneed_array(array: NDArray) -> None:
    """Best-effort MADV_WILLNEED on the mmap backing ``array``.

    Tells the kernel we'll touch the whole int8 cache soon and to retain
    those pages. Without this, per-block "uploading raw int8 genotypes to
    GPU" runs at disk speed instead of RAM speed when pages get evicted
    between training blocks under memory pressure.

    Prefers ``mmap.mmap.madvise`` (works on Linux AND macOS in CPython
    3.8+) and falls back to ``os.posix_madvise`` (Linux-only; missing on
    darwin). No-op on systems without madvise support.
    """
    try:
        import mmap as _mmap_module
    except ImportError:
        return
    base: object = array
    while getattr(base, "base", None) is not None:
        next_base = getattr(base, "base", None)
        if next_base is None:
            break
        base = next_base
        if isinstance(base, _mmap_module.mmap):
            break
    if not isinstance(base, _mmap_module.mmap):
        return

    # Preferred path: mmap.mmap.madvise(MADV_WILLNEED). Available on Linux
    # and macOS; reaches the kernel hint directly without a syscall wrapper
    # mismatch.
    mmap_madvise = getattr(base, "madvise", None)
    mmap_willneed = getattr(_mmap_module, "MADV_WILLNEED", None)
    if mmap_madvise is not None and mmap_willneed is not None:
        try:
            mmap_madvise(mmap_willneed)
            return
        except (OSError, ValueError):
            pass  # fall through to posix_madvise fallback

    # Fallback for older/non-standard runtimes: os.posix_madvise (Linux only;
    # absent on darwin, so it is genuinely a fallback rather than primary).
    posix_madvise = getattr(os, "posix_madvise", None)
    willneed = getattr(os, "POSIX_MADV_WILLNEED", None)
    if posix_madvise is None or willneed is None:
        return
    try:
        posix_madvise(base, 0, len(base), willneed)
    except (OSError, ValueError):
        pass


def as_raw_genotype_matrix(genotypes: RawGenotypeMatrix | NDArray) -> RawGenotypeMatrix:
    if isinstance(genotypes, RawGenotypeMatrix):
        return genotypes
    array = np.asanyarray(genotypes)
    if array.dtype == np.int8:
        return Int8RawGenotypeMatrix(array)
    return DenseRawGenotypeMatrix(np.asarray(array, dtype=np.float32))


def _int8_npy_header_bytes(shape: tuple[int, int], *, fortran_order: bool) -> bytes:
    header_buffer = io.BytesIO()
    write_header: Any = np.lib.format.write_array_header_2_0
    dtype_to_descr: Any = np.lib.format.dtype_to_descr
    write_header(
        header_buffer,
        {
            "descr": dtype_to_descr(np.dtype(np.int8)),
            "fortran_order": bool(fortran_order),
            "shape": tuple(int(dimension) for dimension in shape),
        },
    )
    return header_buffer.getvalue()


def _int8_npy_expected_size(shape: tuple[int, int], *, fortran_order: bool) -> int:
    return len(_int8_npy_header_bytes(shape, fortran_order=fortran_order)) + int(np.prod(shape, dtype=np.int64))


def _has_sufficient_free_space_for_int8_npy(path: Path, shape: tuple[int, int], *, fortran_order: bool) -> tuple[bool, int, int]:
    required_bytes = _int8_npy_expected_size(shape, fortran_order=fortran_order)
    available_bytes = shutil.disk_usage(path).free
    reserve_bytes = max(REDUCED_INT8_CACHE_FREE_SPACE_RESERVE_BYTES, required_bytes // 20)
    return available_bytes >= required_bytes + reserve_bytes, required_bytes, available_bytes


def _stream_write_int8_npy(
    path: Path,
    *,
    shape: tuple[int, int],
    column_batches: Iterator[NDArray],
    fortran_order: bool,
    resume_from_variants: int = 0,
) -> None:
    """Stream int8 column batches to an .npy file, resumable on kill.

    Atomicity model: writes are aimed at ``path`` directly (no random temp
    dir — the caller manages the final rename if needed). After each batch
    the file is fsynced AND a sidecar file at ``str(path) + ".progress"``
    is updated atomically with the variant-count completed so far. On a
    subsequent invocation the caller can pass ``resume_from_variants=N``
    to skip writing the first N variants (the iterator must already have
    skipped those — see the matching caller plumbing in
    ``try_cache_persistently``).

    The header is written once at startup when ``resume_from_variants == 0``;
    on resume the file is opened r+b and we seek straight to the resume
    offset, so the existing header bytes stay valid.
    """
    expected_sample_count = int(shape[0])
    expected_variant_count = int(shape[1])
    if not 0 <= resume_from_variants <= expected_variant_count:
        raise ValueError(
            f"resume_from_variants={resume_from_variants} out of range "
            f"[0, {expected_variant_count}]"
        )
    header_bytes = _int8_npy_header_bytes(shape, fortran_order=fortran_order)
    header_size = len(header_bytes)
    bytes_per_variant = expected_sample_count  # int8 = 1 byte per cell, F-order
    progress_path = path.with_suffix(path.suffix + ".progress")

    if resume_from_variants > 0:
        # Resume: file must exist and be large enough to hold what we claim.
        handle = path.open("r+b")
        handle.seek(header_size + resume_from_variants * bytes_per_variant)
    else:
        handle = path.open("wb")
        handle.write(header_bytes)
        handle.flush()
        os.fsync(handle.fileno())

    written_variant_count = resume_from_variants
    try:
        for batch_values in column_batches:
            batch_array = np.asarray(batch_values, dtype=np.int8)
            if batch_array.ndim != 2:
                raise ValueError("int8 cache batches must be two-dimensional.")
            if batch_array.shape[0] != expected_sample_count:
                raise ValueError(
                    f"int8 cache batch sample count mismatch: {batch_array.shape[0]} != {expected_sample_count}"
                )
            handle.write(np.asfortranarray(batch_array).tobytes(order="F"))
            handle.flush()
            os.fsync(handle.fileno())
            written_variant_count += int(batch_array.shape[1])
            # Atomic per-batch progress write so kill-mid-write loses at
            # most ONE batch's worth of work on the next resume.
            progress_tmp = progress_path.with_suffix(".progress.tmp")
            try:
                progress_tmp.write_text(str(written_variant_count))
                os.replace(progress_tmp, progress_path)
            except OSError:
                # Best-effort: if we can't write the sidecar (read-only fs,
                # quota, etc.) the resume just won't be able to skip and
                # will redo the work — still correct.
                try:
                    progress_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    finally:
        handle.close()
    if written_variant_count != expected_variant_count:
        raise ValueError(
            f"int8 cache variant count mismatch after streaming write: {written_variant_count} != {expected_variant_count}"
        )
    # Final-success cleanup: progress sidecar no longer needed.
    try:
        progress_path.unlink(missing_ok=True)
    except OSError:
        pass


def _scan_int8_npy_resume_point(
    partial_path: Path,
    *,
    shape: tuple[int, int],
    fortran_order: bool,
) -> int:
    """Return how many variants have already been streamed to ``partial_path``.

    Returns 0 when ``partial_path`` doesn't exist, has the wrong header,
    has no progress sidecar, or the sidecar disagrees with the file size.
    A non-zero return guarantees the first N variants in the file are
    valid bytes-on-disk that we can safely skip rewriting.
    """
    if not partial_path.exists():
        return 0
    progress_path = partial_path.with_suffix(partial_path.suffix + ".progress")
    if not progress_path.exists():
        return 0
    try:
        recorded = int(progress_path.read_text().strip())
    except (OSError, ValueError):
        return 0
    expected_sample_count = int(shape[0])
    expected_variant_count = int(shape[1])
    if not 0 < recorded <= expected_variant_count:
        return 0
    try:
        header_bytes = _int8_npy_header_bytes(shape, fortran_order=fortran_order)
    except (ValueError, OSError):
        return 0
    expected_size = len(header_bytes) + recorded * expected_sample_count
    try:
        actual_size = partial_path.stat().st_size
    except OSError:
        return 0
    if actual_size < expected_size:
        return 0
    # Verify the header on disk matches what we'd write — guards against a
    # stale partial from a different shape / dtype / layout.
    try:
        with partial_path.open("rb") as handle:
            on_disk_header = handle.read(len(header_bytes))
    except OSError:
        return 0
    if on_disk_header != header_bytes:
        return 0
    return recorded


@dataclass(slots=True)
class RawGenotypeBatch:
    variant_indices: NDArray
    values: NDArray


class RawGenotypeMatrix(ABC):
    """Abstract base for genotype matrices (samples x variants).

    Values are dosages: 0 = homozygous reference, 1 = heterozygous,
    2 = homozygous alternate, NaN = missing.  Subclasses handle different
    storage backends (in-memory numpy array vs on-disk PLINK .bed file).

    All access is through streaming iterators (iter_column_batches) that
    read a few hundred variants at a time, keeping memory bounded even for
    biobank-scale data (e.g. 447k samples x 900k variants).
    """
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        raise NotImplementedError

    @abstractmethod
    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        raise NotImplementedError

    def standardized(
        self,
        means: NDArray,
        scales: NDArray,
        support_counts: NDArray | None = None,
    ) -> StandardizedGenotypeMatrix:
        return StandardizedGenotypeMatrix(
            raw=self,
            means=np.asarray(means, dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            variant_indices=np.arange(self.shape[1], dtype=np.int32),
            support_counts=None if support_counts is None else np.asarray(support_counts, dtype=np.int32),
        )

    def __array__(self, dtype: np.dtype[Any] | type | None = None) -> NDArray:
        matrix = self.materialize()
        if dtype is None:
            return matrix
        return np.asarray(matrix, dtype=dtype)


class Int8BatchCapable(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...
    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]: ...
    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        *,
        num_threads: int | None = None,
    ) -> Iterator[RawGenotypeBatch]: ...


def _supports_int8_batches(matrix: object) -> TypeGuard[Int8BatchCapable]:
    return hasattr(matrix, "iter_column_batches_i8")


@dataclass(slots=True)
class DenseRawGenotypeMatrix(RawGenotypeMatrix):
    matrix: NDArray

    def __post_init__(self) -> None:
        matrix_array = np.asanyarray(self.matrix)
        if matrix_array.dtype == np.int8:
            self.matrix = matrix_array  # preserve memmap-backed int8 arrays
        else:
            self.matrix = np.asarray(matrix_array, dtype=np.float32)
        if self.matrix.ndim != 2:
            raise ValueError("genotypes must be 2D.")

    def _to_float32(self, batch: NDArray) -> NDArray:
        """Convert a column slice to float32, replacing missing sentinels with NaN."""
        if self.matrix.dtype == np.int8:
            result = batch.astype(np.float32)
            result[batch == PLINK_MISSING_INT8] = np.nan
            return result
        return np.asarray(batch, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=self._to_float32(self.matrix[:, batch_indices]),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        return self._to_float32(self.matrix[:, resolved_indices])


@dataclass(slots=True)
class Int8RawGenotypeMatrix(RawGenotypeMatrix):
    matrix: NDArray

    def __post_init__(self) -> None:
        matrix_array = np.asanyarray(self.matrix)
        self.matrix = matrix_array if matrix_array.dtype == np.int8 else np.asarray(matrix_array, dtype=np.int8)
        if self.matrix.ndim != 2:
            raise ValueError("genotypes must be 2D.")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        *,
        num_threads: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        del num_threads  # in-memory matrix; no decode threads to budget
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            column_index = _contiguous_index_or_slice(batch_indices)
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=np.asarray(self.matrix[:, column_index], dtype=np.int8),
            )

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        for batch in self.iter_column_batches_i8(variant_indices, batch_size=batch_size):
            yield RawGenotypeBatch(
                variant_indices=batch.variant_indices,
                values=_int8_batch_to_float32(batch.values),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        column_index = _contiguous_index_or_slice(resolved_indices)
        return _int8_batch_to_float32(self.matrix[:, column_index])


# Optional re-export of the device-resident bitpacked matrix. The import is
# guarded so that hosts without ``cupy`` (or with a partially-installed CUDA
# stack) can still ``import sv_pgs.genotype``. Callers should test
# ``BitpackedDeviceMatrix is not None`` before using it.
try:  # pragma: no cover - exercised only when cupy / bitpacked stack present
    from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix  # noqa: F401
except Exception:  # noqa: BLE001 - any import-time failure should degrade gracefully
    BitpackedDeviceMatrix = None  # type: ignore[assignment,misc]


@dataclass(slots=True)
class IndexedRawGenotypeMatrix(RawGenotypeMatrix):
    """Expose only a selected subset of a child matrix's columns.

    Used by the multi-VCF dataset loader to drop cross-source duplicate
    variants without rewriting on-disk caches: the wrapper advertises
    shape (n_samples, len(selected_columns)) and routes column i to the
    child's column selected_columns[i]. Yielded batch.variant_indices stay
    in the wrapper's local coordinate space so downstream consumers can
    index back into us directly.
    """
    child: RawGenotypeMatrix
    selected_columns: NDArray

    def __post_init__(self) -> None:
        indices = np.asarray(self.selected_columns, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("selected_columns must be 1D.")
        child_variant_count = int(self.child.shape[1])
        if indices.size and (indices.min() < 0 or indices.max() >= child_variant_count):
            raise ValueError("selected_columns contains an out-of-range index.")
        self.selected_columns = indices

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.child.shape[0]), int(self.selected_columns.shape[0])

    def _child_columns(self, local_indices: NDArray) -> NDArray:
        return self.selected_columns[local_indices].astype(np.int32, copy=False)

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            local_batch = resolved_indices[start_index : start_index + safe_batch_size]
            yield RawGenotypeBatch(
                variant_indices=local_batch,
                values=self.child.materialize(self._child_columns(local_batch)),
            )

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        *,
        num_threads: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        if not _supports_int8_batches(self.child):
            raise RuntimeError("int8 batch iteration requires the wrapped child to support iter_column_batches_i8.")
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        child = self.child
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            local_batch = resolved_indices[start_index : start_index + safe_batch_size]
            child_batch_indices = self._child_columns(local_batch)
            # Read the entire local batch as one child request so columns come
            # back in our coordinate order — no concat or reordering needed.
            child_batch = next(child.iter_column_batches_i8(
                child_batch_indices,
                batch_size=max(child_batch_indices.shape[0], 1),
                num_threads=num_threads,
            ))
            yield RawGenotypeBatch(
                variant_indices=local_batch,
                values=np.asarray(child_batch.values, dtype=np.int8),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        return self.child.materialize(self._child_columns(resolved_indices))


@dataclass(slots=True)
class RowSubsetRawGenotypeMatrix(RawGenotypeMatrix):
    """Lazily expose a row-subset (sample-subset) of a child matrix.

    Replaces the upfront write-temp-mmap reindex path that the multi-VCF /
    multi-source dataset loader previously used. The earlier approach
    materialized a brand-new (n_kept_samples, n_variants) int8 mmap on disk
    per chromosome — for the 80/20 split on the 97k-sample AoU SV cohort
    this writes ~11 GB to /tmp per chromosome × 22+ chromosomes, dominating
    end-to-end wall time (>1 hour just on row reindex) and creating page-
    cache pressure visible as a 25 GB host-RSS spike.

    This wrapper instead holds the original full-sample child + a row index
    array and applies the row selection at iteration time. Downstream uses
    `iter_column_batches[_i8]` exclusively — each batch fetches its
    columns from the child once (memory-mapped) and applies the row
    indexing in a single fancy-indexed numpy slice on the resulting
    contiguous block. No intermediate disk write, no page-cache pressure.

    For repeated passes the kernel page cache covers the hot columns
    naturally, so the lazy form is as fast or faster than the eager form
    on any storage tier where the temp mmap is not faster than the source
    cache. On AoU/GCP they live on the same disk, so this is always a win.
    """
    child: RawGenotypeMatrix
    row_indices: NDArray

    def __post_init__(self) -> None:
        indices = np.asarray(self.row_indices, dtype=np.intp)
        if indices.ndim != 1:
            raise ValueError("row_indices must be 1D.")
        child_row_count = int(self.child.shape[0])
        if indices.size and (indices.min() < 0 or indices.max() >= child_row_count):
            raise ValueError("row_indices contains an out-of-range row.")
        self.row_indices = indices

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.row_indices.shape[0]), int(self.child.shape[1])

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        for batch in self.child.iter_column_batches(variant_indices=variant_indices, batch_size=batch_size):
            yield RawGenotypeBatch(
                variant_indices=batch.variant_indices,
                values=np.ascontiguousarray(batch.values[self.row_indices, :]),
            )

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        *,
        num_threads: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        if not _supports_int8_batches(self.child):
            raise RuntimeError("int8 batch iteration requires the wrapped child to support iter_column_batches_i8.")
        child = self.child
        for batch in child.iter_column_batches_i8(
            variant_indices=variant_indices,
            batch_size=batch_size,
            num_threads=num_threads,
        ):
            yield RawGenotypeBatch(
                variant_indices=batch.variant_indices,
                values=np.ascontiguousarray(batch.values[self.row_indices, :]),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        full = self.child.materialize(variant_indices)
        return np.ascontiguousarray(full[self.row_indices, :])


@dataclass(slots=True)
class PlinkRawGenotypeMatrix(RawGenotypeMatrix):
    bed_path: Path
    sample_indices: NDArray
    variant_count: int
    total_sample_count: int
    batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE
    _reader: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.sample_indices = np.asarray(self.sample_indices, dtype=np.intp)
        if self.sample_indices.ndim != 1:
            raise ValueError("sample_indices must be 1D.")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.sample_indices.shape[0]), int(self.variant_count)

    def _read_batch(self, reader: Any, batch_indices: NDArray) -> NDArray:
        """Read one batch as int8, convert to float32 with NaN for missing."""
        raw_i8 = self._read_batch_i8(reader, batch_indices)
        result = np.asarray(raw_i8, dtype=np.float32)
        result[raw_i8 == PLINK_MISSING_INT8] = np.nan
        return result

    def _read_batch_i8(
        self,
        reader: Any,
        batch_indices: NDArray,
        *,
        num_threads: int | None = None,
    ) -> NDArray:
        """Read one batch as raw int8 (0/1/2/PLINK_MISSING_INT8). No float conversion.

        ``num_threads`` overrides the default ``PLINK_BED_READER_NUM_THREADS``;
        callers running inside a prefetch executor pass a smaller value so the
        total CPU work (pipeline_depth * decode_threads) does not exceed
        ``os.cpu_count()``.
        """
        import time as _time
        sample_index = _contiguous_index_or_slice(self.sample_indices)
        col_index = _contiguous_index_or_slice(batch_indices)
        resolved_threads = (
            PLINK_BED_READER_NUM_THREADS if num_threads is None else max(1, int(num_threads))
        )
        t0 = _time.monotonic()
        result = np.asarray(
            reader.read(
                index=(sample_index, col_index),
                dtype="int8",
                order="F",
                num_threads=resolved_threads,
            ),
            dtype=np.int8,
        )
        elapsed = _time.monotonic() - t0
        # Per-call timing: lets variant-stats logs distinguish bed-decode
        # time from JAX-compute time. Log only when noticeably non-trivial
        # so per-variant gather calls (variant_count == 1) don't spam.
        if elapsed >= 0.5 or int(batch_indices.shape[0]) >= 256:
            log(
                f"      bed read: variants={int(batch_indices.shape[0])} "
                f"samples={int(self.sample_indices.shape[0])}/{int(self.total_sample_count)}  "
                f"elapsed={elapsed:.2f}s"
            )
        return result

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int | None = None,
        *,
        num_threads: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        """Iterate as int8 batches (4x less memory, no float conversion).

        Values are 0/1/2/PLINK_MISSING_INT8 (missing). Callers must handle the shared sentinel.
        """
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        # int8 reads are 4x smaller than float32, but JAX kernels still expand to
        # float32 intermediates (~10 bytes/element peak), so keep decoded batches
        # near 1 GB rather than using the generic float32 reader cap.
        requested = max(int(self.batch_size if batch_size is None else batch_size), 1)
        bytes_per_variant = self.shape[0]  # 1 byte per sample for int8
        max_variants = max(
            PLINK_INT8_TARGET_BATCH_BYTES // max(bytes_per_variant, 1),
            MIN_BED_READER_BATCH_SIZE,
        )
        safe_batch_size = min(requested, max_variants)
        reader = self._bed_reader()
        total = resolved_indices.shape[0]
        batch_mb = self.shape[0] * safe_batch_size / (1024 * 1024)
        n_batches = (total + safe_batch_size - 1) // safe_batch_size
        _log_int8_batch_throttled(
            matrix_id=id(self),
            cache_key=str(self.bed_path),
            batch_size=int(safe_batch_size),
            sample_count=int(self.shape[0]),
            batch_mb=batch_mb,
            n_batches=int(n_batches),
        )

        if total <= safe_batch_size:
            values = self._read_batch_i8(reader, resolved_indices, num_threads=num_threads)
            yield RawGenotypeBatch(variant_indices=resolved_indices, values=values)
            return

        # Keep one read in flight so CPU decode can overlap with JAX stats.
        # Deeper prefetch would launch multiple large PLINK decodes at once,
        # which competes with the reader's own worker threads and doubles
        # decoded-batch memory pressure.
        batch_index_list = [
            resolved_indices[s : s + safe_batch_size]
            for s in range(0, total, safe_batch_size)
        ]
        prefetch_depth = min(PLINK_INT8_MAX_PREFETCH_DEPTH, len(batch_index_list))
        log(f"    int8 prefetch depth={prefetch_depth} (~{batch_mb * prefetch_depth:.0f} MB queued)  mem={mem()}")
        # Split CPU budget between pipeline depth and per-read decode threads so
        # workers * threads_per_worker <= cpu_count and they don't oversubscribe.
        cpu_budget = max(1, os.cpu_count() or 1)
        per_worker_threads = (
            max(1, cpu_budget // prefetch_depth) if num_threads is None else int(num_threads)
        )
        with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
            in_flight: deque[Future[NDArray]] = deque(
                executor.submit(
                    self._read_batch_i8,
                    reader,
                    batch_index_list[i],
                    num_threads=per_worker_threads,
                )
                for i in range(prefetch_depth)
            )
            next_to_submit = prefetch_depth
            for i in range(len(batch_index_list)):
                values = in_flight.popleft().result()
                if next_to_submit < len(batch_index_list):
                    in_flight.append(
                        executor.submit(
                            self._read_batch_i8,
                            reader,
                            batch_index_list[next_to_submit],
                            num_threads=per_worker_threads,
                        )
                    )
                    next_to_submit += 1
                yield RawGenotypeBatch(variant_indices=batch_index_list[i], values=values)

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        requested_batch_size = max(int(self.batch_size if batch_size is None else batch_size), 1)
        safe_batch_size = _effective_bed_reader_batch_size(
            sample_count=self.shape[0],
            requested_batch_size=requested_batch_size,
        )
        reader = self._bed_reader()
        total = resolved_indices.shape[0]

        # Build list of batch index arrays
        batch_index_list: list[NDArray] = []
        for start_index in range(0, total, safe_batch_size):
            batch_index_list.append(resolved_indices[start_index : start_index + safe_batch_size])

        if len(batch_index_list) <= 1:
            # Single batch — no prefetch overhead needed
            for batch_indices in batch_index_list:
                values = self._read_batch(reader, batch_indices)
                yield RawGenotypeBatch(variant_indices=batch_indices, values=values)
            return

        # Prefetch: read batch N+1 in background thread while caller processes batch N
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit first read
            future = executor.submit(self._read_batch, reader, batch_index_list[0])
            for i in range(len(batch_index_list)):
                # Wait for current batch
                values = future.result()
                # Submit next batch read (if any) before yielding
                if i + 1 < len(batch_index_list):
                    future = executor.submit(self._read_batch, reader, batch_index_list[i + 1])
                yield RawGenotypeBatch(variant_indices=batch_index_list[i], values=values)

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        reader = self._bed_reader()
        return self._read_batch(reader, resolved_indices)

    def _bed_reader(self) -> Any:
        # Bind to a local FIRST so a concurrent ``release_reader`` (running on
        # another thread between batches to drop the cached mmap so fadvise
        # can evict page-cache bytes) can't null out ``self._reader`` between
        # our open and our return. Previously returning ``self._reader``
        # directly after assigning it raced with release_reader resetting the
        # attribute to None and returned None to callers, which then crashed
        # with ``'NoneType' object has no attribute 'read'`` on the next
        # transpose_matvec pass of the marginal screen.
        reader = self._reader
        if reader is None:
            log(f"    opening PLINK reader (lazy, no metadata): iid_count={self.total_sample_count} sid_count={self.variant_count}  mem={mem()}")
            reader = open_bed(
                self.bed_path,
                iid_count=self.total_sample_count,
                sid_count=self.variant_count,
                properties={},
                num_threads=None,
            )
            self._reader = reader
            log(f"    PLINK reader opened  mem={mem()}")
        return reader

    def release_reader(self) -> None:
        """Drop the cached PLINK reader so its fd/mmap can be reclaimed.

        Required between PLINK batches when the host runs under a cgroup
        memory limit: ``posix_fadvise(POSIX_FADV_DONTNEED)`` on the .bed
        file cannot evict pages whose PTE map_count > 0 (mmap'd readers)
        or that are still referenced by an active fd's read-ahead state,
        so we must drop our reference and let __del__ tear it down before
        fadvise runs.

        Thread-safe: in-flight reads on another thread hold a *local*
        reference to the reader (grabbed at the top of
        :meth:`iter_column_batches_i8`), so clearing ``self._reader``
        only nullifies the cache attribute. The active read continues
        with its local handle. Once that local reference drops, refcount
        hits zero and the reader's __del__ closes the fd. We deliberately
        do NOT call ``.close()`` here — that would yank the fd out from
        under any in-flight read and raise
        ``RuntimeError: PLINK reader file descriptor is closed``.
        """
        self._reader = None


@dataclass(slots=True)
class ConcatenatedRawGenotypeMatrix(RawGenotypeMatrix):
    children: tuple[RawGenotypeMatrix, ...]
    _sample_count: int = field(init=False, repr=False)
    _variant_offsets: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("children cannot be empty.")
        self._sample_count = int(self.children[0].shape[0])
        variant_offsets = [0]
        for child in self.children:
            if int(child.shape[0]) != self._sample_count:
                raise ValueError("all concatenated genotype matrices must have the same sample count.")
            variant_offsets.append(variant_offsets[-1] + int(child.shape[1]))
        self._variant_offsets = np.asarray(variant_offsets, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, int]:
        return self._sample_count, int(self._variant_offsets[-1])

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            child_ids = np.searchsorted(self._variant_offsets[1:], batch_indices, side="right")
            batch_values = np.empty((self.shape[0], batch_indices.shape[0]), dtype=np.float32)
            for child_index in np.unique(child_ids):
                child_positions = np.nonzero(child_ids == child_index)[0]
                child_variant_indices = batch_indices[child_positions] - int(self._variant_offsets[child_index])
                batch_values[:, child_positions] = self.children[int(child_index)].materialize(child_variant_indices)
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=batch_values,
            )

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        *,
        num_threads: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        if not all(_supports_int8_batches(child) for child in self.children):
            raise RuntimeError("int8 batch iteration requires every concatenated child to support iter_column_batches_i8.")
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            child_ids = np.searchsorted(self._variant_offsets[1:], batch_indices, side="right")
            batch_values = np.empty((self.shape[0], batch_indices.shape[0]), dtype=np.int8)
            for child_index in np.unique(child_ids):
                child_positions = np.nonzero(child_ids == child_index)[0]
                child_variant_indices = batch_indices[child_positions] - int(self._variant_offsets[child_index])
                child = cast(Int8BatchCapable, self.children[int(child_index)])
                child_batch_iter = child.iter_column_batches_i8(
                    child_variant_indices,
                    batch_size=max(child_variant_indices.shape[0], 1),
                    num_threads=num_threads,
                )
                child_batch = next(child_batch_iter)
                batch_values[:, child_positions] = np.asarray(child_batch.values, dtype=np.int8)
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=batch_values,
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        child_ids = np.searchsorted(self._variant_offsets[1:], resolved_indices, side="right")
        matrix = np.empty((self.shape[0], resolved_indices.shape[0]), dtype=np.float32)
        for child_index in np.unique(child_ids):
            child_positions = np.nonzero(child_ids == child_index)[0]
            child_variant_indices = resolved_indices[child_positions] - int(self._variant_offsets[child_index])
            matrix[:, child_positions] = self.children[int(child_index)].materialize(child_variant_indices)
        return matrix

    def _all_bitpacked(self) -> bool:
        if BitpackedDeviceMatrix is None:
            return False
        return all(isinstance(child, BitpackedDeviceMatrix) for child in self.children)

    def matvec_numpy(self, x_np: NDArray) -> NDArray:
        """Dispatch G @ x to bitpacked child kernels when every child is bitpacked."""
        if not self._all_bitpacked():
            raise NotImplementedError("matvec_numpy requires all children to be BitpackedDeviceMatrix.")
        x = np.asarray(x_np, dtype=np.float32).ravel()
        if x.shape[0] != int(self._variant_offsets[-1]):
            raise ValueError(f"matvec_numpy: x has shape {x.shape}, expected ({int(self._variant_offsets[-1])},).")
        out = np.zeros((self._sample_count,), dtype=np.float32)
        for child_index, child in enumerate(self.children):
            start = int(self._variant_offsets[child_index])
            stop = int(self._variant_offsets[child_index + 1])
            out += child.matvec_numpy(x[start:stop])
        return out

    def rmatvec_numpy(self, y_np: NDArray) -> NDArray:
        """Dispatch G.T @ y to bitpacked child kernels when every child is bitpacked."""
        if not self._all_bitpacked():
            raise NotImplementedError("rmatvec_numpy requires all children to be BitpackedDeviceMatrix.")
        y = np.asarray(y_np, dtype=np.float32).ravel()
        if y.shape[0] != self._sample_count:
            raise ValueError(f"rmatvec_numpy: y has shape {y.shape}, expected ({self._sample_count},).")
        out = np.empty((int(self._variant_offsets[-1]),), dtype=np.float32)
        for child_index, child in enumerate(self.children):
            start = int(self._variant_offsets[child_index])
            stop = int(self._variant_offsets[child_index + 1])
            out[start:stop] = child.rmatvec_numpy(y)
        return out

    def gram_block(self, variant_indices: NDArray) -> Any:
        """Subset gram dispatched onto bitpacked children when uniform.

        Falls back to ``NotImplementedError`` for mixed children; callers
        should drop back to the int8 streaming path in that case.
        """
        if not self._all_bitpacked():
            raise NotImplementedError("gram_block requires all children to be BitpackedDeviceMatrix.")
        idx = np.asarray(variant_indices, dtype=np.int64).ravel()
        child_ids = np.searchsorted(self._variant_offsets[1:], idx, side="right")
        # If the requested subset falls entirely within one child, dispatch
        # directly. Cross-child gram blocks aren't supported by per-child
        # kernels alone, so fall back to NotImplementedError.
        unique_ids = np.unique(child_ids)
        if unique_ids.shape[0] != 1:
            raise NotImplementedError("gram_block across multiple bitpacked children not supported.")
        child_index = int(unique_ids[0])
        local = idx - int(self._variant_offsets[child_index])
        return self.children[child_index].gram_block(local)


_cupy_module = None
_cupy_checked = False


def _try_import_cupy() -> Any | None:
    """Import CuPy, caching the result. Returns None only during tests."""
    global _cupy_module, _cupy_checked
    if _cupy_checked:
        return _cupy_module
    _cupy_checked = True
    try:
        import cupy
        if _cupy_runtime_usable(cupy):
            _cupy_module = cupy
            return cupy
    except (ImportError, OSError, RuntimeError):
        pass
    _cupy_module = None
    return None


_libc_malloc_trim: Any = None
_libc_malloc_trim_checked = False


def _release_plink_readers(raw: Any) -> int:
    """Call release_reader() on every PlinkRawGenotypeMatrix in the tree.

    This is the only thing that lets ``posix_fadvise(POSIX_FADV_DONTNEED)``
    actually evict bed pages from the page cache — fadvise skips pages whose
    PTEs are still held by an active mmap, and bed-reader keeps its mmap
    open for the lifetime of the Bed object. Dropping the cached reader
    breaks the mmap and lets the kernel reclaim those pages.

    Returns the number of readers released so the caller can log it.
    """
    released = 0
    seen: set[int] = set()
    stack: list[Any] = [raw]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        release = getattr(node, "release_reader", None)
        if callable(release) and getattr(node, "bed_path", None) is not None:
            try:
                release()
                released += 1
            except (OSError, AttributeError, RuntimeError):
                pass
            continue
        child = getattr(node, "child", None)
        if child is not None:
            stack.append(child)
        children = getattr(node, "children", None)
        if children is not None:
            stack.extend(children)
    return released


def _has_plink_backing(raw: Any) -> bool:
    """True if ``raw`` (or any wrapped child/children) carries a bed_path."""
    seen: set[int] = set()
    stack: list[Any] = [raw]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if getattr(node, "bed_path", None) is not None:
            return True
        child = getattr(node, "child", None)
        if child is not None:
            stack.append(child)
        children = getattr(node, "children", None)
        if children is not None:
            stack.extend(children)
    return False


def _collect_bed_paths(raw: Any) -> tuple[str, ...]:
    """Walk wrapper genotype matrices to collect every underlying .bed path.

    The standardized view used by the marginal screen is typically a
    ``RowSubsetRawGenotypeMatrix(child=PlinkRawGenotypeMatrix(...))`` —
    ``getattr(raw, "bed_path", None)`` on the wrapper returns None and the
    fadvise(DONTNEED) silently no-ops, defeating the page-cache eviction
    that keeps the AoU cgroup happy. Descend through ``.child`` and
    ``.children`` to find every PlinkRawGenotypeMatrix in the tree.
    """
    seen: set[int] = set()
    collected: list[str] = []
    stack: list[Any] = [raw]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        bed_path = getattr(node, "bed_path", None)
        if bed_path is not None:
            collected.append(str(bed_path))
            continue  # leaf
        child = getattr(node, "child", None)
        if child is not None:
            stack.append(child)
        children = getattr(node, "children", None)
        if children is not None:
            for sub in children:
                stack.append(sub)
    return tuple(collected)


def _fadvise_dontneed(path: str | Path) -> None:
    """Tell the kernel to drop page-cache pages for ``path``.

    bed-reader mmaps the .bed file and the kernel caches every page touched
    by ``reader.read(...)`` in the page cache. On a 30 GB AoU container the
    cgroup memory accounting includes file-backed cache, so a marginal screen
    that walks the whole .bed (~107 GB of bit-packed bytes for 956 k variants
    × 447 k samples) drives the cgroup OOM-killer long before the *anonymous*
    process RSS would warrant it. ``POSIX_FADV_DONTNEED`` is the only way to
    proactively evict those pages — they are shared with bed-reader's mmap,
    and the kernel reclaims them lazily under pressure, but ``free_all_blocks``
    and ``malloc_trim`` do nothing about them.
    """
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        posix_fadvise(fd, 0, 0, dontneed)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _release_host_caches(
    cupy: Any | None,
    *,
    fadvise_paths: Sequence[str | Path] = (),
) -> None:
    """Return per-batch host scratch back to the OS.

    Three retention paths grow RSS monotonically across PLINK batches on the
    30 GB AoU container and together push it past the cgroup limit:

    1. cupy's default ``PinnedMemoryPool`` holds the staging buffer used by
       every ``cupy.asarray(numpy_array)`` H2D copy. Even with ``blocking=True``
       the pool keeps the largest block per stream alive forever.
    2. glibc's main arena does not auto-trim after large freed allocations.
       ``malloc_trim(0)`` walks the arena and ``munmap``s freed pages.
    3. **The dominant one on AoU.** bed-reader mmaps the .bed file and the
       kernel page-caches every read. Cgroup memory accounting includes file
       cache, so the marginal screen blows the limit before the screen
       finishes. ``POSIX_FADV_DONTNEED`` on the .bed file evicts those pages.

    All three calls are cheap and idempotent.

    NOTE: previously this also called ``gc.collect()`` to force bed-reader's
    mmap to tear down before fadvise. That triggered JAX's
    ``_xla_gc_callback`` which walks every device buffer in the process —
    with a multi-GB CuPy pool the gc pass took minutes per batch. Killed
    it. ``_release_plink_readers`` already nulls the cached reader
    attribute; refcount-based teardown handles the rest. fadvise is
    best-effort either way; if the kernel can't evict a few pages this
    iteration it will next time pressure rises.
    """
    for path in fadvise_paths:
        _fadvise_dontneed(path)
    if cupy is not None:
        try:
            pinned_pool = cupy.get_default_pinned_memory_pool()
        except (AttributeError, *_cupy_runtime_error_classes(cupy)):
            pinned_pool = None
        if pinned_pool is not None:
            try:
                pinned_pool.free_all_blocks()
            except _cupy_runtime_error_classes(cupy):
                pass
    global _libc_malloc_trim, _libc_malloc_trim_checked
    if not _libc_malloc_trim_checked:
        _libc_malloc_trim_checked = True
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=False)
            trim = getattr(libc, "malloc_trim", None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                _libc_malloc_trim = trim
        except (OSError, AttributeError):
            _libc_malloc_trim = None
    if _libc_malloc_trim is not None:
        try:
            _libc_malloc_trim(0)
        except OSError:
            pass


def _cupy_device_count(cupy: Any) -> int:
    try:
        return max(int(cupy.cuda.runtime.getDeviceCount()), 0)
    except _cupy_runtime_error_classes(cupy):
        return 1


def _cupy_device_ids(cupy: Any) -> tuple[int, ...]:
    return tuple(range(_cupy_device_count(cupy)))


@contextmanager
def _cupy_device_context(cupy: Any, device_id: int) -> Iterator[None]:
    device_factory = getattr(getattr(cupy, "cuda", None), "Device", None)
    if device_factory is None:
        yield
        return
    try:
        device = device_factory(int(device_id))
    except TypeError:
        device = device_factory()
    enter = getattr(device, "__enter__", None)
    exit_method = getattr(device, "__exit__", None)
    if enter is None or exit_method is None:
        device.use()
        yield
        return
    enter()
    try:
        yield
    finally:
        exit_method(None, None, None)


def _cupy_device_synchronize(cupy: Any, device_id: int) -> None:
    with _cupy_device_context(cupy, device_id):
        try:
            cupy.cuda.Device().synchronize()
        except _cupy_runtime_error_classes(cupy):
            pass


def _cupy_current_device_id(cupy: Any) -> int:
    device_factory = getattr(getattr(cupy, "cuda", None), "Device", None)
    if device_factory is None:
        return 0
    try:
        return int(device_factory().id)
    except _cupy_runtime_error_classes(cupy):
        return 0


def _split_contiguous_columns(total_columns: int, device_ids: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    cols = int(total_columns)
    if cols < 0:
        raise ValueError("total_columns must be non-negative.")
    active_devices = tuple(int(device_id) for device_id in device_ids)
    if not active_devices:
        return ()
    shard_count = min(len(active_devices), cols) if cols > 0 else 1
    splits: list[tuple[int, int, int]] = []
    start = 0
    for shard_index, device_id in enumerate(active_devices[:shard_count]):
        remaining_columns = cols - start
        remaining_shards = shard_count - shard_index
        width = remaining_columns // remaining_shards
        stop = start + width
        splits.append((device_id, start, stop))
        start = stop
    return tuple(splits)


# Cache of (stream_a, stream_b) keyed by the id of the cupy module so that
# tests with mocked cupy instances each get their own pair, while real runs
# create the two non-default streams exactly once per process.
_cuda_upload_streams_cache: dict[tuple[int, int], tuple[Any, Any]] = {}


def _cuda_upload_stream_pair(cupy: Any) -> tuple[Any, Any]:
    """Return two non-default CUDA streams used to overlap H2D copies with CPU work.

    Created once per (cupy module) and reused across every materialization.
    """
    cache_key = (id(cupy), _cupy_current_device_id(cupy))
    cached = _cuda_upload_streams_cache.get(cache_key)
    if cached is not None:
        return cached
    stream_a = cupy.cuda.Stream(non_blocking=True)
    stream_b = cupy.cuda.Stream(non_blocking=True)
    pair = (stream_a, stream_b)
    _cuda_upload_streams_cache[cache_key] = pair
    return pair


def _pinned_int8_host_buffer(
    cupy: Any,
    sample_count: int,
    max_tile_variants: int,
) -> tuple[NDArray, NDArray, Any]:
    """Allocate a pair of reusable page-locked int8 host buffers (one per stream).

    Returns ``(buffer_a, buffer_b, pinned_memory_owner)``; each buffer is shaped
    ``(sample_count, max_tile_variants)``. The two buffers share a single
    pinned-memory allocation drawn from the process-wide pinned buffer pool
    (see ``sv_pgs.bitpacked_loader._PinnedBufferPool``) so they live as long
    as the returned owner reference. Pass the owner to
    ``_release_pinned_int8_host_buffer`` once the upload streams have
    synchronized to return it to the pool — the next upload pass will
    reuse the pin instead of round-tripping cudaHostAlloc/cudaFreeHost.

    Raises ``MemoryError``/``RuntimeError`` loudly if the CUDA driver cannot
    pin host memory — there is no silent fallback.
    """
    if int(sample_count) <= 0 or int(max_tile_variants) <= 0:
        raise ValueError(
            "pinned int8 host buffer requires positive sample_count and max_tile_variants; "
            f"got sample_count={sample_count}, max_tile_variants={max_tile_variants}."
        )
    slot_shape = (int(sample_count), int(max_tile_variants))
    slot_element_count = slot_shape[0] * slot_shape[1]
    total_nbytes = slot_element_count * 2 * np.dtype(np.int8).itemsize
    # Acquire through the shared pool; reinterpret the uint8 backing buffer
    # as int8 since the caller fills/uses the slots as int8.
    from sv_pgs.bitpacked_loader import _allocate_pinned as _acquire_pinned

    pinned_memory, _u8_view = _acquire_pinned(cupy, total_nbytes)
    flat_view = np.frombuffer(pinned_memory, dtype=np.int8, count=slot_element_count * 2)
    buffer_a = flat_view[:slot_element_count].reshape(slot_shape)
    buffer_b = flat_view[slot_element_count:].reshape(slot_shape)
    return buffer_a, buffer_b, pinned_memory


def _release_pinned_int8_host_buffer(pinned_memory: Any) -> None:
    """Return a pinned int8 host buffer to the process-wide pool.

    Safe with ``None`` (no-op) and on double-release. Callers should invoke
    after every in-flight upload event has synchronized so the host buffer
    is not referenced by any pending DMA.
    """
    if pinned_memory is None:
        return
    from sv_pgs.bitpacked_loader import _release_pinned

    _release_pinned(pinned_memory)


def _upload_standardized_int8_tiles_overlapped(
    *,
    cupy: Any,
    raw_int8: "Int8BatchCapable",
    variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    gpu_destination: Any,
    sample_count: int,
    upload_batch_size: int,
    standardized_dtype: Any,
) -> None:
    """Upload + standardize int8 tiles into a float GPU strip with overlapped H2D.

    Mirrors ``_upload_int8_tiles_overlapped`` but additionally casts each tile to
    ``standardized_dtype`` on the GPU and applies ``(x - mean) / scale`` with the
    PLINK missing-int8 sentinel zeroed out, matching ``_standardize_batch_cupy``
    bit-for-bit while keeping the work on the upload stream so it overlaps the
    next tile's BED-reader pass.
    """
    stream_pair = _cuda_upload_stream_pair(cupy)
    pinned_buffer_a, pinned_buffer_b, _pinned_owner = _pinned_int8_host_buffer(
        cupy, sample_count=sample_count, max_tile_variants=int(upload_batch_size)
    )
    pinned_slots = (pinned_buffer_a, pinned_buffer_b)
    in_flight_events: list[Any] = [None, None]
    selected_means_gpu = cupy.asarray(means[variant_indices], dtype=standardized_dtype)
    selected_scales_gpu = cupy.asarray(scales[variant_indices], dtype=standardized_dtype)
    local_start = 0
    for tile_index, raw_batch in enumerate(
        raw_int8.iter_column_batches_i8(variant_indices, batch_size=upload_batch_size)
    ):
        slot_index = tile_index % 2
        previous_event = in_flight_events[slot_index]
        if previous_event is not None:
            previous_event.synchronize()
        batch_width = raw_batch.values.shape[1]
        pinned_slot = pinned_slots[slot_index]
        pinned_slot[:, :batch_width] = raw_batch.values
        local_stop = local_start + batch_width
        stream = stream_pair[slot_index]
        with stream:
            staged_int8 = cupy.asarray(pinned_slot[:, :batch_width])
            standardized_tile = _standardize_int8_cupy(
                staged_int8,
                selected_means_gpu[local_start:local_stop],
                selected_scales_gpu[local_start:local_stop],
                cupy,
                dtype=standardized_dtype,
            )
            gpu_destination[:, local_start:local_stop] = standardized_tile
        in_flight_events[slot_index] = stream.record()
        local_start = local_stop
    # Synchronization invariant: after this function returns, the caller assigns
    # ``gpu_destination`` to ``self._cupy_cache`` and downstream consumers will
    # immediately read from it. We must therefore wait on every in-flight async
    # H2D event — in particular the LAST tile's event, which would otherwise
    # still be in flight when the loop body exits.
    for in_flight_event in in_flight_events:
        if in_flight_event is not None:
            in_flight_event.synchronize()
    # All H2D events have completed; return the pinned int8 host buffer
    # to the process-wide pool so the next upload (SNP+SV path, or the
    # next disease in the same process) reuses the pin instead of
    # round-tripping cudaHostAlloc/cudaFreeHost.
    _release_pinned_int8_host_buffer(_pinned_owner)


def _try_upload_int8_parallel_memmap(
    *,
    cupy: Any,
    raw: object,
    variant_indices: NDArray,
    gpu_destination: Any,
    sample_count: int,
    n_workers: int = 8,
) -> bool:
    """Fast parallel-memmap upload path for an F-order int8 numpy memmap leaf.

    Bypasses the per-batch iterator chain: allocates one big pinned host buffer,
    splits variant_indices into ``n_workers`` column stripes, then for each stripe
    a worker thread (a) reads the memmap slice into its stripe of the pinned
    buffer — letting the kernel issue parallel disk I/O across worker threads —
    and (b) issues an async H2D copy on its own non-blocking CUDA stream into the
    matching GPU column range. Returns True on success; False if the source isn't
    eligible (no memmap leaf, wrong dtype/order, or non-contiguous indices).

    Bit-identical to ``_upload_int8_tiles_overlapped`` byte-for-byte: same raw
    int8 bytes get DMA'd to the same GPU offsets.
    """
    if not isinstance(raw, Int8RawGenotypeMatrix):
        return False
    matrix = raw.matrix
    if not isinstance(matrix, np.memmap):
        return False
    if matrix.dtype != np.int8:
        return False
    if matrix.ndim != 2:
        return False
    if not matrix.flags.f_contiguous:
        return False
    if int(matrix.shape[0]) != int(sample_count):
        return False
    n_variants = int(variant_indices.shape[0])
    if n_variants == 0:
        return True
    vi = np.asarray(variant_indices, dtype=np.int64)
    if vi.size > 1:
        if not np.all(np.diff(vi) == 1):
            return False
    src_col_start = int(vi[0])
    src_col_end = src_col_start + n_variants
    if src_col_start < 0 or src_col_end > int(matrix.shape[1]):
        return False

    total_bytes = int(sample_count) * n_variants
    try:
        # Go through the process-wide pinned pool so the next call with
        # a same-size request reuses this allocation rather than re-
        # pinning. ``alloc_pinned_memory`` itself is what we want to
        # avoid hammering on every disease iteration.
        from sv_pgs.bitpacked_loader import _allocate_pinned as _acquire_pinned

        pinned_owner, _u8_view = _acquire_pinned(cupy, total_bytes)
    except (MemoryError, RuntimeError) as exc:
        log(f"    parallel-pread upload: pinned alloc failed ({exc}); falling back")
        return False
    flat = np.frombuffer(pinned_owner, dtype=np.int8, count=total_bytes)
    pinned_buf = flat.reshape((n_variants, sample_count)).T
    if not pinned_buf.flags.f_contiguous:
        return False

    try:
        if matrix.filename:
            with open(matrix.filename, "rb") as fadvise_handle:
                file_offset = int(getattr(matrix, "offset", 0)) + src_col_start * int(sample_count)
                # posix_fadvise is Linux-only; fall through cleanly on macOS/Windows.
                posix_fadvise = getattr(os, "posix_fadvise", None)
                willneed = getattr(os, "POSIX_FADV_WILLNEED", None)
                if posix_fadvise is not None and willneed is not None:
                    try:
                        posix_fadvise(fadvise_handle.fileno(), file_offset, total_bytes, willneed)
                    except OSError:
                        pass
    except (AttributeError, OSError):
        pass

    effective_workers = max(1, min(int(n_workers), n_variants))
    stripe_size = (n_variants + effective_workers - 1) // effective_workers
    stripes: list[tuple[int, int]] = []
    for worker_idx in range(effective_workers):
        s = worker_idx * stripe_size
        e = min(s + stripe_size, n_variants)
        if s < e:
            stripes.append((s, e))
    effective_workers = len(stripes)

    streams = [cupy.cuda.Stream(non_blocking=True) for _ in range(effective_workers)]
    events: list[Any] = [None] * effective_workers
    errors: list[BaseException] = []

    runtime = getattr(getattr(cupy, "cuda", None), "runtime", None)
    memcpy_async = getattr(runtime, "memcpyAsync", None) if runtime is not None else None
    memcpy_h2d_kind = getattr(runtime, "memcpyHostToDevice", None) if runtime is not None else None
    direct_h2d_supported = (
        memcpy_async is not None
        and memcpy_h2d_kind is not None
        and hasattr(gpu_destination, "data")
        and hasattr(getattr(gpu_destination, "data", None), "ptr")
    )

    def worker(worker_idx: int, col_start: int, col_end: int) -> None:
        try:
            src_view = matrix[:, src_col_start + col_start : src_col_start + col_end]
            # Forces parallel page faults + memcpy from page cache → pinned buffer.
            # numpy releases the GIL for this memcpy, so workers fan out across cores.
            pinned_buf[:, col_start:col_end] = src_view
            stripe_bytes = int(sample_count) * (col_end - col_start)
            stream = streams[worker_idx]
            if direct_h2d_supported:
                # Direct async H2D into the pre-allocated GPU destination slice.
                # The naïve ``cupy.asarray(pinned_slice)`` route would allocate
                # ``stripe_bytes`` of GPU staging per worker — at AoU sizes that
                # is ~1.5 GB × 8 workers ≈ 12 GB on top of the 11.8 GB
                # destination, OOM'ing a 15 GB T4. ``memcpyAsync`` writes
                # straight into the F-order destination view: zero extra GPU
                # allocation, identical bytes on-device.
                assert memcpy_async is not None and memcpy_h2d_kind is not None
                gpu_view = gpu_destination[:, col_start:col_end]
                host_slice = pinned_buf[:, col_start:col_end]
                memcpy_async(
                    int(gpu_view.data.ptr),
                    int(host_slice.ctypes.data),
                    int(stripe_bytes),
                    memcpy_h2d_kind,
                    int(stream.ptr),
                )
            else:
                with stream:
                    gpu_int8_dtype = getattr(cupy, "int8", np.int8)
                    staged_tile = cupy.asarray(pinned_buf[:, col_start:col_end], dtype=gpu_int8_dtype)
                    gpu_destination[:, col_start:col_end] = staged_tile
            events[worker_idx] = stream.record()
        except BaseException as exc:
            errors.append(exc)

    log(
        "    parallel-pread upload: "
        + f"{n_variants} variants x {sample_count} samples ({total_bytes / 1e9:.1f} GB) "
        + f"across {effective_workers} workers/streams  mem={mem()}"
    )
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="i8-pread") as executor:
        futures = [executor.submit(worker, i, s, e) for i, (s, e) in enumerate(stripes)]
        for future in futures:
            future.result()

    if errors:
        raise errors[0]

    for event in events:
        if event is not None:
            event.synchronize()
    # All async H2D copies have landed — return the pinned staging buffer
    # to the pool for reuse by the next parallel-pread upload pass.
    _release_pinned_int8_host_buffer(pinned_owner)
    return True


def _upload_int8_tiles_overlapped(
    *,
    cupy: Any,
    raw_int8: "Int8BatchCapable",
    variant_indices: NDArray,
    gpu_destination: Any,
    sample_count: int,
    upload_batch_size: int,
    gpu_int8_dtype: Any,
) -> None:
    """Upload int8 column tiles into ``gpu_destination`` with overlapped H2D copies.

    Issues each tile's async H2D copy on one of two CUDA streams, double-buffered
    through a pair of pinned int8 host buffers so the next tile's BED-reader
    CPU work runs concurrently with the previous tile's H2D transfer. The math
    is bit identical to a serial pageable ``cupy.asarray`` upload — this is a
    pure scheduling change.
    """
    stream_pair = _cuda_upload_stream_pair(cupy)
    pinned_buffer_a, pinned_buffer_b, _pinned_owner = _pinned_int8_host_buffer(
        cupy, sample_count=sample_count, max_tile_variants=int(upload_batch_size)
    )
    pinned_slots = (pinned_buffer_a, pinned_buffer_b)
    in_flight_events: list[Any] = [None, None]
    local_start = 0
    for tile_index, raw_batch in enumerate(
        raw_int8.iter_column_batches_i8(variant_indices, batch_size=upload_batch_size)
    ):
        slot_index = tile_index % 2
        previous_event = in_flight_events[slot_index]
        if previous_event is not None:
            previous_event.synchronize()
        batch_width = raw_batch.values.shape[1]
        pinned_slot = pinned_slots[slot_index]
        pinned_slot[:, :batch_width] = raw_batch.values
        local_stop = local_start + batch_width
        stream = stream_pair[slot_index]
        with stream:
            staged_tile = cupy.asarray(pinned_slot[:, :batch_width], dtype=gpu_int8_dtype)
            gpu_destination[:, local_start:local_stop] = staged_tile
        in_flight_events[slot_index] = stream.record()
        local_start = local_stop
    # Synchronization invariant: after this function returns, the caller assigns
    # ``gpu_destination`` to ``self._cupy_cache`` and downstream consumers will
    # immediately read from it. We must therefore wait on every in-flight async
    # H2D event — in particular the LAST tile's event, which would otherwise
    # still be in flight when the loop body exits.
    for in_flight_event in in_flight_events:
        if in_flight_event is not None:
            in_flight_event.synchronize()
    # All H2D events have completed; return the pinned int8 host buffer
    # to the process-wide pool for reuse by the next upload pass.
    _release_pinned_int8_host_buffer(_pinned_owner)


_gpu_verified = False


def require_gpu() -> Any:
    """Probe GPU+CuPy at pipeline entry and log the selected runtime."""
    global _gpu_verified
    if _gpu_verified:
        return _cupy_module
    from sv_pgs.progress import log

    cupy = _try_import_cupy()
    if cupy is None:
        log("  CUDA runtime diagnostic: " + _cupy_runtime_diagnostic())
        log("  NVIDIA driver diagnostic: " + _nvidia_driver_diagnostic())
        log("  no usable NVIDIA GPU detected — running CPU-only (this will be slow)")
        _gpu_verified = True
        return None
    # Reclaim any pool blocks before sampling free memory so the warning
    # below reflects real availability rather than blocks the CuPy pool
    # is merely caching.
    _release_cupy_cached_memory(cupy)
    device_count = max(_cupy_device_count(cupy), 1)
    free_bytes = 0
    total_bytes = 0
    per_device = []
    for device_id in range(device_count):
        with _cupy_device_context(cupy, device_id):
            device_free, device_total = cupy.cuda.runtime.memGetInfo()
        free_bytes += int(device_free)
        total_bytes += int(device_total)
        per_device.append(f"device{device_id}={int(device_free) / 1e9:.1f}/{int(device_total) / 1e9:.1f}GB")
    try:
        from sv_pgs._jax import SELECTED_CUDA_DEVICE as _selected_cuda_device
    except (ImportError, RuntimeError):
        _selected_cuda_device = None
    pinned_suffix = ""
    if _selected_cuda_device is not None:
        pinned_suffix = f" pinned={_selected_cuda_device[0]}"
    log(
        "  GPU verified: "
        + f"cuda_devices={device_count}{pinned_suffix} aggregate={total_bytes / 1e9:.1f} GB total, {free_bytes / 1e9:.1f} GB free "
        + "(" + ", ".join(per_device) + ")"
    )
    # If another process is pinning most of the device, the training pipeline
    # will silently fall back to slow streaming paths and may still OOM on the
    # remaining residual. Surface it once, at entry, so the user can kill stale
    # kernels before burning hours on a degraded run.
    if total_bytes > 0 and free_bytes < total_bytes * 0.5:
        held_bytes = total_bytes - free_bytes
        log(
            f"  WARNING: only {free_bytes / 1e9:.1f} GB of {total_bytes / 1e9:.1f} GB free "
            f"({held_bytes / 1e9:.1f} GB held by other processes). "
            f"Run `nvidia-smi` and kill stale GPU processes to avoid OOM / slow streaming fallbacks."
        )
    # Report CuPy default mempool limit + the process-wide pinned buffer
    # pool state at startup. The mempool limit governs HBM fragmentation
    # behaviour for the bitpacked / dense paths; the pinned pool stat
    # makes it observable whether subsequent loads are hitting reuse
    # (n_reuses > 0) or paying the full pin cost each time.
    try:
        pool = cupy.get_default_memory_pool()
        get_limit = getattr(pool, "get_limit", None)
        limit_bytes = int(get_limit()) if get_limit is not None else 0
        if limit_bytes <= 0:
            limit_repr = "unbounded"
        else:
            limit_repr = f"{limit_bytes / 1e9:.2f} GB"
        try:
            from sv_pgs.bitpacked_loader import _pinned_pool

            ps = _pinned_pool().stats()
            pinned_repr = (
                f"pinned-pool[allocs={ps['allocs']} reuses={ps['reuses']} "
                f"avail={ps['available_count']}@{ps['available_bytes'] / 1e9:.2f}GB]"
            )
        except (ImportError, RuntimeError, AttributeError):
            pinned_repr = "pinned-pool[unavailable]"
        log(f"  GPU mempool limit={limit_repr}  {pinned_repr}")
    except (AttributeError, RuntimeError, OSError):
        # Logging is best-effort; never let a probe failure take down
        # require_gpu().
        pass
    _gpu_verified = True
    return cupy


def _as_gpu_compute_jax(array: Any) -> JaxArray:
    return jnp.asarray(array, dtype=gpu_compute_jax_dtype())


def _cupy_to_jax(array: Any) -> JaxArray:
    """Convert CuPy result to JAX, preferring zero-copy DLPack interop."""
    if hasattr(array, "__dlpack__"):
        return jax_dlpack.from_dlpack(array).astype(gpu_compute_jax_dtype())
    return jnp.asarray(array.get(), dtype=gpu_compute_jax_dtype())


def _to_cupy_float32(array: Any) -> Any:
    """Convert JAX/numpy array to CuPy float32 for CuPy matmul."""
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    if type(array).__module__.startswith("cupy"):
        return array.astype(cupy.float32, copy=False)
    if isinstance(array, jax.Array) and hasattr(cupy, "from_dlpack"):
        return cupy.from_dlpack(array).astype(cupy.float32, copy=False)
    return cupy.asarray(np.asarray(array, dtype=np.float32))


def _to_cupy_float64(array: Any) -> Any:
    """Convert JAX/numpy array to CuPy float64 for numerically sensitive GPU solves."""
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    if type(array).__module__.startswith("cupy"):
        return array.astype(cupy.float64, copy=False)
    if isinstance(array, jax.Array) and hasattr(cupy, "from_dlpack"):
        return cupy.from_dlpack(array).astype(cupy.float64, copy=False)
    return cupy.asarray(np.asarray(array, dtype=np.float64))


def _cupy_compute_dtype(cupy: Any) -> Any:
    return cupy.float32 if gpu_compute_numpy_dtype() == np.dtype(np.float32) else cupy.float64


def _cupy_dtype_to_numpy_dtype(dtype: Any) -> np.dtype[Any]:
    try:
        return np.dtype(dtype)
    except TypeError:
        return np.dtype(np.float32)


def _standardize_int8_cupy(raw_values: Any, means: Any, scales: Any, cupy: Any, *, dtype: Any) -> Any:
    """Standardize int8 genotypes on GPU without materializing a mask buffer."""
    resolved_dtype = cupy.float32 if dtype is None else dtype
    elementwise_kernel = getattr(cupy, "ElementwiseKernel", None)
    if elementwise_kernel is not None:
        kernel_cache = getattr(_standardize_int8_cupy, "_kernel_cache", None)
        if kernel_cache is None:
            kernel_cache = {}
            setattr(_standardize_int8_cupy, "_kernel_cache", kernel_cache)
        cache_key = id(elementwise_kernel)
        kernel = kernel_cache.get(cache_key)
        if kernel is None:
            kernel = elementwise_kernel(
                "int8 raw, T means, T scales, int8 missing",
                "T out",
                # NVRTC can't pick between (T)0 (float16) and the rhs of the
                # ternary (also T) when T is __half because __half ↔ float
                # both have implicit conversions. Compute both arms as T
                # explicitly with an intermediate, then select.
                "T _val = ((T)raw - means) / scales; out = (raw == missing) ? (T)0 : _val",
                "sv_pgs_standardize_int8_missing_zero",
            )
            kernel_cache[cache_key] = kernel
        return kernel(raw_values, means[None, :], scales[None, :], np.int8(PLINK_MISSING_INT8))

    standardized = raw_values.astype(resolved_dtype, copy=False)
    valid_mask = standardized != float(PLINK_MISSING_INT8)
    standardized -= means[None, :]
    standardized /= scales[None, :]
    multiply = getattr(cupy, "multiply", np.multiply)
    multiply(standardized, valid_mask, out=standardized)
    return standardized


def _standardize_int8_cupy_into(raw_values: Any, means: Any, scales: Any, output: Any, cupy: Any, *, dtype: Any) -> Any:
    """Standardize int8 genotypes into a caller-owned fp staging buffer."""
    resolved_dtype = cupy.float32 if dtype is None else dtype
    elementwise_kernel = getattr(cupy, "ElementwiseKernel", None)
    if elementwise_kernel is not None:
        kernel_cache = getattr(_standardize_int8_cupy_into, "_kernel_cache", None)
        if kernel_cache is None:
            kernel_cache = {}
            setattr(_standardize_int8_cupy_into, "_kernel_cache", kernel_cache)
        cache_key = id(elementwise_kernel)
        kernel = kernel_cache.get(cache_key)
        if kernel is None:
            kernel = elementwise_kernel(
                "int8 raw, T means, T scales, int8 missing",
                "T out",
                # NVRTC can't pick between (T)0 (float16) and the rhs of the
                # ternary (also T) when T is __half because __half ↔ float
                # both have implicit conversions. Compute both arms as T
                # explicitly with an intermediate, then select.
                "T _val = ((T)raw - means) / scales; out = (raw == missing) ? (T)0 : _val",
                "sv_pgs_standardize_int8_missing_zero_into",
            )
            kernel_cache[cache_key] = kernel
        return kernel(raw_values, means[None, :], scales[None, :], np.int8(PLINK_MISSING_INT8), output)

    output[...] = raw_values
    valid_mask = output != float(PLINK_MISSING_INT8)
    output -= means[None, :]
    output /= scales[None, :]
    multiply = getattr(cupy, "multiply", np.multiply)
    multiply(output, valid_mask, out=output)
    return output


def _to_cupy_compute(array: Any) -> Any:
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    compute_dtype = _cupy_compute_dtype(cupy)
    if compute_dtype == cupy.float32:
        return _to_cupy_float32(array)
    return _to_cupy_float64(array)


def _cupy_to_numpy(array: Any, *, dtype: Any) -> NDArray:
    host_array = array.get() if hasattr(array, "get") else array
    return np.asarray(host_array, dtype=dtype)


def _standardize_batch_cupy(
    batch_values: NDArray,
    means: Any,
    scales: Any,
    cupy: Any,
    *,
    missing_sentinel: int | None = None,
    dtype: Any = None,
) -> Any:
    """Standardize a raw batch directly on GPU.

    Memory-sensitive: boolean-mask scatter (``standardized[missing] = 0``) is
    implemented in CuPy via a prefix-sum scan that allocates a batch-sized
    scratch buffer, and ``cupy.where`` allocates a second batch-sized float
    buffer; either OOMs on wide AoU batches. The int8 path uses a fused
    elementwise kernel; the float path uses in-place NaN scrubbing.
    """
    resolved_dtype = cupy.float32 if dtype is None else dtype
    if missing_sentinel is not None:
        raw_gpu = cupy.asarray(batch_values, dtype=getattr(cupy, "int8", np.int8))
        return _standardize_int8_cupy(raw_gpu, means, scales, cupy, dtype=resolved_dtype)

    standardized = cupy.asarray(batch_values, dtype=resolved_dtype)
    # Raw float source: missing entries arrive as NaN and propagate as NaN
    # through center/scale. ``NaN * 0 == NaN`` (IEEE 754), so a multiply-by-
    # mask would not clear them — use in-place nan_to_num instead, which
    # also scrubs ±inf from zero-scale columns at no extra cost.
    standardized -= means[None, :]
    standardized /= scales[None, :]
    nan_to_num = getattr(cupy, "nan_to_num", None)
    if nan_to_num is not None:
        nan_to_num(standardized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        host = np.asarray(standardized)
        np.nan_to_num(host, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        standardized = cupy.asarray(host, dtype=resolved_dtype)
    return standardized


def _iter_standardized_gpu_batches(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    variant_indices: NDArray,
    means: Any,
    scales: Any,
    *,
    batch_size: int,
    cupy: Any,
    dtype: Any = None,
) -> Iterator[tuple[slice, Any]]:
    resolved_dtype = cupy.float32 if dtype is None else dtype
    selected_means = cupy.asarray(means[variant_indices], dtype=resolved_dtype)
    selected_scales = cupy.asarray(scales[variant_indices], dtype=resolved_dtype)
    for batch_slice, values in _iter_prefetched_raw_batches(raw, variant_indices, batch_size=batch_size):
        for relative_slice, standardized_batch in _iter_standardized_gpu_subbatches(
            values,
            selected_means[batch_slice],
            selected_scales[batch_slice],
            cupy,
            missing_sentinel=int(PLINK_MISSING_INT8) if _supports_int8_batches(raw) else None,
            dtype=resolved_dtype,
        ):
            yield (
                slice(
                    int(batch_slice.start or 0) + int(relative_slice.start or 0),
                    int(batch_slice.start or 0) + int(relative_slice.stop or 0),
                ),
                standardized_batch,
            )


def _iter_standardized_gpu_subbatches(
    values: NDArray,
    means: Any,
    scales: Any,
    cupy: Any,
    *,
    missing_sentinel: int | None,
    dtype: Any,
) -> Iterator[tuple[slice, Any]]:
    try:
        yield slice(0, int(values.shape[1])), _standardize_batch_cupy(
            values,
            means,
            scales,
            cupy,
            missing_sentinel=missing_sentinel,
            dtype=dtype,
        )
        return
    except Exception as exc:
        if not _is_cupy_out_of_memory(exc) or int(values.shape[1]) <= 1:
            raise
        _release_cupy_cached_memory(cupy)

    split_at = max(int(values.shape[1]) // 2, 1)
    log(
        "        CuPy OOM while standardizing "
        + f"{int(values.shape[1]):,} variants; retrying as {split_at:,}"
        + f" + {int(values.shape[1]) - split_at:,} variants  mem={mem()}"
    )
    for child_slice, standardized_batch in _iter_standardized_gpu_subbatches(
        values[:, :split_at],
        means[:split_at],
        scales[:split_at],
        cupy,
        missing_sentinel=missing_sentinel,
        dtype=dtype,
    ):
        yield child_slice, standardized_batch
    for child_slice, standardized_batch in _iter_standardized_gpu_subbatches(
        values[:, split_at:],
        means[split_at:],
        scales[split_at:],
        cupy,
        missing_sentinel=missing_sentinel,
        dtype=dtype,
    ):
        yield (
            slice(
                split_at + int(child_slice.start or 0),
                split_at + int(child_slice.stop or 0),
            ),
            standardized_batch,
        )


def _iter_prefetched_raw_batches(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    variant_indices: NDArray,
    *,
    batch_size: int,
    io_worker_limit: int | None = None,
    max_prefetch_bytes: int | None = None,
    decode_thread_limit: int | None = None,
    read_semaphore: threading.Semaphore | None = None,
) -> Iterator[tuple[slice, NDArray]]:
    variant_indices_arr = np.asarray(variant_indices)
    n = int(variant_indices_arr.shape[0])
    if n == 0:
        return
    safe_batch_size = max(int(batch_size), 1)
    sample_count = int(raw.shape[0])

    chunk_starts = list(range(0, n, safe_batch_size))
    chunks = [variant_indices_arr[s : s + safe_batch_size] for s in chunk_starts]

    # Parallel disk reads help on cloud SSDs, but large AoU row subsets make
    # each chunk hundreds of MB. Bound both workers and queued chunks by bytes
    # so prefetch does not evict the pages/GPU buffers the current matmul needs.
    bytes_per_raw_value = np.dtype(np.int8 if _supports_int8_batches(raw) else np.float32).itemsize
    chunk_bytes = max(sample_count * safe_batch_size * bytes_per_raw_value, 1)
    prefetch_target_bytes = (
        GPU_STANDARDIZED_PREFETCH_TARGET_BYTES
        if max_prefetch_bytes is None
        else max(1, int(max_prefetch_bytes))
    )
    memory_capped_in_flight = max(1, prefetch_target_bytes // chunk_bytes)
    # Reserve 2 cores: one for the main consumer thread and one for the CUDA
    # host driver / GPU command submission. The remaining cores fan out across
    # the prefetch executor. Previously the cap was the raw cpu_count, which on
    # the AoU 16-core T4 box left the CUDA driver thread fighting prefetch
    # workers for CPU time and starving the GPU.
    cpu_cap = max(1, (os.cpu_count() or 4) - 2)
    worker_cap = cpu_cap if io_worker_limit is None else max(1, min(cpu_cap, int(io_worker_limit)))
    num_io_workers = min(worker_cap, int(memory_capped_in_flight), max(1, len(chunks)))
    # Allow up to 4x as many decoded chunks in flight as we have workers so the
    # GPU never waits for a decode to finish; the byte-budget cap above still
    # bounds the total queued bytes.
    in_flight_limit = min(num_io_workers * 4, int(memory_capped_in_flight), len(chunks))
    # Split the CPU budget between pipeline depth and per-read decode fan-out.
    # The pipeline executor *is* the parallelism; fanning out further inside
    # each read would oversubscribe (workers * threads_per_worker > cpu_count)
    # and pile up per-thread numpy scratch until the process swaps.
    if decode_thread_limit is None:
        per_worker_decode_threads = max(1, cpu_cap // max(1, num_io_workers))
    else:
        per_worker_decode_threads = max(1, int(decode_thread_limit))

    if _supports_int8_batches(raw):
        i8_raw = raw
        def _read_chunk(chunk_indices: NDArray) -> NDArray:
            if read_semaphore is None:
                for batch in i8_raw.iter_column_batches_i8(
                    chunk_indices,
                    batch_size=chunk_indices.shape[0],
                    num_threads=per_worker_decode_threads,
                ):
                    return np.asarray(batch.values)
            else:
                with read_semaphore:
                    for batch in i8_raw.iter_column_batches_i8(
                        chunk_indices,
                        batch_size=chunk_indices.shape[0],
                        num_threads=per_worker_decode_threads,
                    ):
                        return np.asarray(batch.values)
            return np.empty((sample_count, 0), dtype=np.int8)
    else:
        float_raw = raw
        def _read_chunk(chunk_indices: NDArray) -> NDArray:
            if read_semaphore is None:
                for batch in float_raw.iter_column_batches(chunk_indices, batch_size=chunk_indices.shape[0]):
                    return np.asarray(batch.values)
            else:
                with read_semaphore:
                    for batch in float_raw.iter_column_batches(chunk_indices, batch_size=chunk_indices.shape[0]):
                        return np.asarray(batch.values)
            return np.empty((sample_count, 0), dtype=np.float32)
    if num_io_workers <= 1 or len(chunks) <= 1:
        local_start = 0
        for chunk in chunks:
            values = _read_chunk(chunk)
            batch_width = values.shape[1]
            local_stop = local_start + batch_width
            yield slice(local_start, local_stop), values
            del values
            local_start = local_stop
        return

    executor = ThreadPoolExecutor(
        max_workers=num_io_workers,
        thread_name_prefix="standardized-gpu-prefetch",
    )
    futures: deque[Future[Any]] = deque()
    next_to_submit = 0

    def _submit_more() -> None:
        nonlocal next_to_submit
        while next_to_submit < len(chunks) and len(futures) < in_flight_limit:
            futures.append(executor.submit(_read_chunk, chunks[next_to_submit]))
            next_to_submit += 1

    try:
        _submit_more()
        local_start = 0
        while futures:
            future = futures.popleft()
            values = future.result()
            _submit_more()
            batch_width = values.shape[1]
            local_stop = local_start + batch_width
            yield slice(local_start, local_stop), values
            del values
            local_start = local_stop
    finally:
        # wait=True so worker threads finish releasing the large decoded arrays
        # they hold before this generator returns; wait=False was leaking
        # workers across iterations and pinning their per-thread allocations.
        for pending in futures:
            pending.cancel()
        futures.clear()
        executor.shutdown(wait=True, cancel_futures=True)


# =========================================================================
# GPU-side PLINK BED 2-bit decoder (default path for Plink-backed inputs).
#
# Replaces the CPU decode + 1 GB host int8 buffer per batch with:
#   pread raw 2-bit bytes (~256 MB) -> small host transient
#       -> cupy.asarray to GPU
#       -> RawKernel unpacks 2-bit + applies sample subset
#       -> int8 lives on GPU; downstream standardize+matmul unchanged
#
# Bit-exact equivalent of sv_pgs.plink._DECODE_LOOKUP_A1:
#     code 00 -> 2 (hom A1), 01 -> -127 (missing), 10 -> 1 (het), 11 -> 0 (hom A2)
#
# Host transient = bytes_per_variant * batch_variants (~340 MB on AoU) instead
# of 1 GB per batch. The 1 GB int8 array never exists on host. Combined with
# fadvise(DONTNEED) on the .bed fd this makes host-side OOM structurally
# impossible regardless of variant count, and the 2-bit unpack runs in
# microseconds on V100 (vs ~15 s on 3 CPU threads), freeing both the GPU
# and the CPU for other work.
# =========================================================================

_PLINK_GPU_DECODE_KERNEL_SRC = r"""
extern "C" __global__ void plink_decode_a1_subset(
    const unsigned char* __restrict__ packed,    // [n_variants, bytes_per_variant]
    long long bytes_per_variant,
    const long long* __restrict__ sample_indices, // [n_kept_samples]
    long long n_kept_samples,
    long long n_variants,
    signed char* __restrict__ out_fortran        // [n_kept_samples, n_variants] F-order
) {
    long long s = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long v = (long long)blockIdx.y;
    if (s >= n_kept_samples || v >= n_variants) return;
    long long sample_id = sample_indices[s];
    long long byte_idx  = sample_id >> 2;          // /4
    int bit_offset      = ((int)(sample_id & 3LL)) << 1;  // (%4)*2
    unsigned char byte  = packed[v * bytes_per_variant + byte_idx];
    int code            = (byte >> bit_offset) & 3;
    // _DECODE_LOOKUP_A1 = [2, -127, 1, 0]
    signed char val;
    if (code == 0)      val = (signed char)2;
    else if (code == 1) val = (signed char)-127;
    else if (code == 2) val = (signed char)1;
    else                val = (signed char)0;
    out_fortran[v * n_kept_samples + s] = val;
}
"""

_PLINK_DECODE_LOOKUP_A1 = np.array([2, -127, 1, 0], dtype=np.int8)


def _decode_packed_bytes_reference(
    packed: NDArray,
    *,
    bytes_per_variant: int,
    sample_indices: NDArray,
    n_variants: int,
) -> NDArray:
    """CPU reference implementation of the GPU 2-bit decode.

    Used by tests to verify bit-exact equivalence with the GPU kernel.
    Returns an int8 F-order array of shape (n_kept_samples, n_variants).
    """
    packed_u8 = np.asarray(packed, dtype=np.uint8).reshape(int(n_variants), int(bytes_per_variant))
    samples = np.asarray(sample_indices, dtype=np.int64)
    byte_idx = samples >> 2
    bit_offset = (samples & 3).astype(np.int64) << 1
    bytes_sel = packed_u8[:, byte_idx]
    codes = (bytes_sel >> bit_offset[None, :]) & 0x3
    decoded = _PLINK_DECODE_LOOKUP_A1[codes]
    return np.asfortranarray(decoded.T)


def _resolve_plink_pread_context(raw: Any) -> tuple[Any, NDArray, int, str] | None:
    """If ``raw`` is Plink-backed, return (reader, sample_indices, iid_count, bed_path).

    Walks .child to find the leaf PlinkRawGenotypeMatrix. The first
    RowSubsetRawGenotypeMatrix encountered along the way contributes
    sample_indices; if none is present we use the leaf's own sample_indices
    (which for the streaming consumer always covers the kept sample set).
    Returns None for any non-Plink path so the caller can fall back.

    Logs the specific failure mode whenever it returns None so we don't
    silently fall back to the slow CPU-decode path without telling the user
    why — historically this was a debugging black hole (the previous run
    spent 38 min in the CPU path because no log told us the fast path was
    being declined).
    """
    sample_indices: NDArray | None = None
    seen: set[int] = set()
    visited_kinds: list[str] = []
    # DFS so we descend through both single-child wrappers (.child) AND
    # multi-child concatenators (.children). On AoU the runtime wrapper tree
    # is RowSubset(child=Concatenated(children=[Plink])) — the previous
    # version followed only .child and stopped at the concatenator, silently
    # falling back to the slow CPU path. Walk both.
    stack: list[Any] = [raw]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        visited_kinds.append(type(node).__name__)
        if sample_indices is None and getattr(node, "row_indices", None) is not None:
            sample_indices = np.asarray(node.row_indices, dtype=np.int64)
        if getattr(node, "bed_path", None) is not None:
            try:
                reader = node._bed_reader()
            except (OSError, AttributeError, RuntimeError) as exc:
                log(f"    GPU-decode skipped: _bed_reader raised ({exc}) on {type(node).__name__}")
                return None
            if not hasattr(reader, "_pread_indexed_variant_payload"):
                log(
                    f"    GPU-decode skipped: reader {type(reader).__name__} has no "
                    f"_pread_indexed_variant_payload (need sv_pgs.plink.open_bed)"
                )
                return None
            if sample_indices is None:
                leaf_samples = getattr(node, "sample_indices", None)
                if leaf_samples is None:
                    log(f"    GPU-decode skipped: leaf {type(node).__name__} has no sample_indices")
                    return None
                sample_indices = np.asarray(leaf_samples, dtype=np.int64)
            iid_count = int(getattr(node, "total_sample_count", 0))
            if iid_count <= 0:
                log(f"    GPU-decode skipped: total_sample_count={iid_count} on {type(node).__name__}")
                return None
            return reader, sample_indices, iid_count, str(node.bed_path)
        child = getattr(node, "child", None)
        if child is not None:
            stack.append(child)
        children = getattr(node, "children", None)
        if children is not None:
            # If concatenator has multiple Plink leaves, the GPU-direct path
            # can only handle ONE bed_path per call — bail out to the CPU
            # path which already handles concatenation. Single-child
            # concatenators (the common AoU shape) flow through normally.
            children_list = list(children)
            plink_children = [
                c for c in children_list
                if c is not None and getattr(c, "bed_path", None) is not None
            ]
            if len(plink_children) > 1:
                log(
                    f"    GPU-decode skipped: concatenator has {len(plink_children)} "
                    f"Plink leaves; multi-source not yet supported in GPU-decode path"
                )
                return None
            for sub in children_list:
                stack.append(sub)
    log(
        f"    GPU-decode skipped: no bed_path in wrapper chain "
        f"[{' -> '.join(visited_kinds) if visited_kinds else '<empty>'}]; "
        f"falling back to CPU-decode path"
    )
    return None


_plink_gpu_decode_kernel_cache: Any = None


def _get_plink_gpu_decode_kernel(cupy: Any) -> Any:
    global _plink_gpu_decode_kernel_cache
    if _plink_gpu_decode_kernel_cache is None:
        _plink_gpu_decode_kernel_cache = cupy.RawKernel(
            _PLINK_GPU_DECODE_KERNEL_SRC,
            "plink_decode_a1_subset",
        )
    return _plink_gpu_decode_kernel_cache


def _gpu_plink_pread_transpose_matmul_direct(
    *,
    reader: Any,
    sample_indices: NDArray,
    iid_count: int,
    bed_path: str,
    selected_variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    matrix_gpu: Any,
    batch_size: int,
    cupy: Any,
    dtype: Any,
    progress_label: str | None,
) -> Any:
    """Direct pread -> GPU-decode -> standardize -> matmul.

    Replaces the CPU decode path (which allocates a 1 GB int8 numpy array
    per batch) with a small ~340 MB raw-bytes host transient + GPU unpack.
    Host RAM stays bounded regardless of variant count.
    """
    import time as _time

    n_variants_total = int(selected_variant_indices.shape[0])
    n_kept_samples = int(sample_indices.shape[0])
    bytes_per_variant = (int(iid_count) + 3) // 4
    sample_indices_int64 = np.asarray(sample_indices, dtype=np.int64)
    selected_variant_indices_int64 = np.asarray(selected_variant_indices, dtype=np.int64)

    sample_indices_gpu = cupy.asarray(sample_indices_int64)
    selected_means_gpu = cupy.asarray(means[selected_variant_indices], dtype=dtype)
    selected_scales_gpu = cupy.asarray(scales[selected_variant_indices], dtype=dtype)
    result_gpu = cupy.empty(
        (n_variants_total, matrix_gpu.shape[1]), dtype=dtype
    )
    kernel = _get_plink_gpu_decode_kernel(cupy)

    if progress_label is not None:
        log(
            f"        {progress_label}: GPU-decode path active "
            f"(bytes/variant={bytes_per_variant:,}, kept_samples={n_kept_samples:,}, "
            f"batch_size={batch_size}, host_xient~{bytes_per_variant * batch_size / 1e6:.0f} MB)  mem={mem()}"
        )

    t_start = _time.monotonic()
    last_log_time = t_start

    block_dim = 128
    n_batches_total = (n_variants_total + batch_size - 1) // batch_size

    # 1-deep prefetch: a background thread preads batch N+1 while the main
    # thread is busy uploading + decoding + matmul'ing batch N on the GPU.
    # Disk and GPU run concurrently — total wall-time ≈ max(disk, gpu)
    # per batch instead of disk + gpu. On AoU the disk is slow enough that
    # this nearly halves screen time even on one V100.
    def _pread_batch(idx: int, start: int, stop: int) -> tuple[float, NDArray]:
        t = _time.monotonic()
        payload_bytes = reader._pread_indexed_variant_payload(
            selected_variant_indices_int64[start:stop],
            bytes_per_variant=bytes_per_variant,
        )
        return _time.monotonic() - t, payload_bytes

    prefetch_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="plink-pread-prefetch"
    )
    batch_ranges = [
        (i, b_start, min(b_start + batch_size, n_variants_total))
        for i, b_start in enumerate(range(0, n_variants_total, batch_size))
    ]
    # Submit the first batch's pread now so it can overlap with our setup.
    pending: Future[tuple[float, NDArray]] | None = (
        prefetch_executor.submit(_pread_batch, *batch_ranges[0])
        if batch_ranges
        else None
    )
    try:
        for queue_pos, (batch_index, batch_start, batch_stop) in enumerate(batch_ranges):
            assert pending is not None
            pread_seconds, payload = pending.result()
            # Immediately queue the NEXT batch's pread so the disk works
            # while we do the GPU work for this batch.
            if queue_pos + 1 < len(batch_ranges):
                pending = prefetch_executor.submit(_pread_batch, *batch_ranges[queue_pos + 1])
            else:
                pending = None

            n_batch_vars = batch_stop - batch_start

            t_upload = _time.monotonic()
            packed_gpu = cupy.asarray(payload, blocking=True)
            del payload
            upload_seconds = _time.monotonic() - t_upload

            t_gpu = _time.monotonic()
            int8_gpu = cupy.empty((n_kept_samples, n_batch_vars), dtype=cupy.int8, order="F")
            grid_x = (n_kept_samples + block_dim - 1) // block_dim
            grid_y = n_batch_vars
            kernel(
                (grid_x, grid_y),
                (block_dim,),
                (
                    packed_gpu,
                    np.int64(bytes_per_variant),
                    sample_indices_gpu,
                    np.int64(n_kept_samples),
                    np.int64(n_batch_vars),
                    int8_gpu,
                ),
            )
            del packed_gpu

            means_slice = selected_means_gpu[batch_start:batch_stop]
            scales_slice = selected_scales_gpu[batch_start:batch_stop]
            standardized_gpu = _standardize_int8_cupy(
                int8_gpu,
                means_slice,
                scales_slice,
                cupy,
                dtype=dtype,
            )
            result_gpu[batch_start:batch_stop, :] = standardized_gpu.T @ matrix_gpu
            del int8_gpu, standardized_gpu
            gpu_seconds = _time.monotonic() - t_gpu

            # Page-cache eviction on the pread fd. No mmap here so fadvise
            # works cleanly without needing to drop any cached reader.
            _release_host_caches(cupy, fadvise_paths=(bed_path,))

            if progress_label is not None:
                completed = batch_stop
                now = _time.monotonic()
                elapsed = now - t_start
                rate = completed / max(elapsed, 1e-6)
                eta = (n_variants_total - completed) / max(rate, 1e-6)
                log(
                    f"        {progress_label}: batch {batch_index + 1}/{n_batches_total} "
                    f"({completed:,}/{n_variants_total:,} = {100*completed/n_variants_total:.1f}%) "
                    f"pread={pread_seconds:.1f}s upload={upload_seconds:.2f}s gpu={gpu_seconds:.2f}s "
                    f"elapsed={elapsed:.0f}s rate={rate:,.0f}v/s eta={eta:.0f}s  mem={mem()}"
                )
                last_log_time = now
    finally:
        # Cancel any pending prefetch (if we exited the loop via exception)
        # and shut down the executor so its worker thread joins cleanly.
        if pending is not None:
            pending.cancel()
        prefetch_executor.shutdown(wait=True, cancel_futures=True)
    if progress_label is not None:
        elapsed = _time.monotonic() - t_start
        log(
            f"        {progress_label}: GPU-decode done {n_variants_total:,} variants "
            f"in {elapsed:.1f}s ({n_variants_total / max(elapsed, 1e-6):,.0f}v/s)  mem={mem()}"
        )
    return result_gpu


def _gpu_streaming_batch_size(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    *,
    sample_count: int,
    requested_batch_size: int,
    cupy: Any = None,
    dtype: Any = None,
) -> int:
    supports_int8 = _supports_int8_batches(raw)
    if supports_int8:
        requested_batch_size = max(int(requested_batch_size), auto_batch_size_i8(sample_count))
    resolved_cupy = _try_import_cupy() if cupy is None else cupy
    static_target_batch_bytes = (
        GPU_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        if supports_int8
        else GPU_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
    )
    if resolved_cupy is not None:
        target_batch_bytes = _gpu_dynamic_standardized_target_batch_bytes(
            resolved_cupy,
            static_target_batch_bytes=static_target_batch_bytes,
            free_fraction=(
                GPU_INT8_STANDARDIZED_DYNAMIC_FREE_FRACTION
                if supports_int8
                else GPU_STANDARDIZED_DYNAMIC_FREE_FRACTION
            ),
        )
    else:
        target_batch_bytes = static_target_batch_bytes
    return _effective_gpu_standardized_streaming_batch_size(
        sample_count=sample_count,
        requested_batch_size=requested_batch_size,
        target_batch_bytes=target_batch_bytes,
        dtype=dtype,
    )


def _gpu_int8_transpose_matmul(
    *,
    raw_int8: Int8BatchCapable,
    variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    matrix_gpu: Any,
    batch_size: int,
    cupy: Any,
    dtype: Any,
    progress_label: str | None,
) -> Any:
    selected_variant_indices = np.asarray(variant_indices, dtype=np.int32)

    # Multi-GPU sharded path comes FIRST. The sharded function checks the
    # GPU-decode context per shard internally, so we get BOTH multi-GPU
    # parallelism AND the GPU-decode fast path on each V100 at the same
    # time. Previously the GPU-decode check came first and routed straight
    # to the single-device direct path, leaving the second V100 idle for
    # the entire marginal screen.
    device_ids = _cupy_device_ids(cupy)
    if len(device_ids) >= 2 and int(selected_variant_indices.shape[0]) >= len(device_ids):
        return _gpu_int8_transpose_matmul_sharded(
            raw_int8=raw_int8,
            selected_variant_indices=selected_variant_indices,
            means=means,
            scales=scales,
            matrix_gpu=matrix_gpu,
            batch_size=batch_size,
            cupy=cupy,
            dtype=dtype,
            progress_label=progress_label,
            device_ids=device_ids,
        )
    # Single-GPU path: prefer GPU-decode when Plink-backed.
    ctx = _resolve_plink_pread_context(raw_int8)
    if ctx is not None:
        reader, sample_indices, iid_count, bed_path = ctx
        return _gpu_plink_pread_transpose_matmul_direct(
            reader=reader,
            sample_indices=sample_indices,
            iid_count=iid_count,
            bed_path=bed_path,
            selected_variant_indices=selected_variant_indices,
            means=means,
            scales=scales,
            matrix_gpu=matrix_gpu,
            batch_size=batch_size,
            cupy=cupy,
            dtype=dtype,
            progress_label=progress_label,
        )
    return _gpu_int8_transpose_matmul_single_device(
        raw_int8=raw_int8,
        selected_variant_indices=selected_variant_indices,
        means=means,
        scales=scales,
        matrix_gpu=matrix_gpu,
        batch_size=batch_size,
        cupy=cupy,
        dtype=dtype,
        progress_label=progress_label,
        io_worker_limit=None,
    )


def _gpu_int8_transpose_matmul_single_device(
    *,
    raw_int8: Int8BatchCapable,
    selected_variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    matrix_gpu: Any,
    batch_size: int,
    cupy: Any,
    dtype: Any,
    progress_label: str | None,
    io_worker_limit: int | None,
    max_prefetch_bytes: int | None = None,
    decode_thread_limit: int | None = None,
    read_semaphore: threading.Semaphore | None = None,
) -> Any:
    selected_variant_indices = np.asarray(selected_variant_indices, dtype=np.int32)
    selected_means_gpu = cupy.asarray(means[selected_variant_indices], dtype=dtype)
    selected_scales_gpu = cupy.asarray(scales[selected_variant_indices], dtype=dtype)
    result_gpu = cupy.empty((selected_variant_indices.shape[0], matrix_gpu.shape[1]), dtype=dtype)
    total_variants = int(selected_variant_indices.shape[0])
    completed_variants = 0
    last_logged_variants = 0
    log_interval = max(total_variants // 50, 1)
    import time as _time
    t_start = _time.monotonic()
    resolved_fadvise_paths = _collect_bed_paths(raw_int8)
    if progress_label is not None:
        log(
            f"        {progress_label}: start streaming {total_variants:,} variants "
            f"(batch_size={batch_size}, fadvise_bed_files={len(resolved_fadvise_paths)})  "
            f"mem={mem()}"
        )
        # If the underlying raw exposes a bed_path anywhere in its wrapper tree
        # we expect to find it — silent failure here would mean the per-batch
        # fadvise(DONTNEED) drain doesn't fire and the cgroup page cache grows
        # unbounded, exactly the OOM we keep hitting on AoU.
        if not resolved_fadvise_paths and _has_plink_backing(raw_int8):
            log(
                f"        {progress_label}: WARNING — Plink-backed matrix but "
                f"no bed_path resolved via wrapper walk; per-batch page-cache "
                f"eviction will be a no-op (OOM risk)"
            )
    for batch_slice, host_values in _iter_prefetched_raw_batches(
        raw_int8,
        selected_variant_indices,
        batch_size=batch_size,
        io_worker_limit=io_worker_limit,
        max_prefetch_bytes=max_prefetch_bytes,
        decode_thread_limit=decode_thread_limit,
        read_semaphore=read_semaphore,
    ):
        int8_gpu = cupy.asarray(host_values, dtype=getattr(cupy, "int8", np.int8), blocking=True)
        del host_values
        means_gpu = selected_means_gpu[batch_slice]
        scales_gpu = selected_scales_gpu[batch_slice]
        standardized_gpu = _standardize_int8_cupy(
            int8_gpu,
            means_gpu,
            scales_gpu,
            cupy,
            dtype=dtype,
        )
        result_gpu[batch_slice, :] = standardized_gpu.T @ matrix_gpu
        del int8_gpu, standardized_gpu
        # Drop bed-reader's cached mmap before the fadvise pass — otherwise
        # the page-cache eviction is a no-op because POSIX_FADV_DONTNEED
        # skips pages whose PTE map_count > 0. With the mmap gone, gc
        # forced inside _release_host_caches lets __del__ run and fadvise
        # actually evicts the just-decoded bed pages.
        _release_plink_readers(raw_int8)
        _release_host_caches(cupy, fadvise_paths=resolved_fadvise_paths)
        if progress_label is not None:
            completed_variants = batch_slice.stop
            if completed_variants - last_logged_variants >= log_interval:
                last_logged_variants = completed_variants
                elapsed_seconds = _time.monotonic() - t_start
                rate = completed_variants / max(elapsed_seconds, 1e-6)
                eta = (total_variants - completed_variants) / max(rate, 1e-6)
                log(
                    f"        {progress_label}: {completed_variants:,}/{total_variants:,} "
                    f"({100*completed_variants/total_variants:.1f}%) "
                    f"elapsed={elapsed_seconds:.0f}s rate={rate:,.0f}v/s eta={eta:.0f}s  mem={mem()}"
                )
    if progress_label is not None:
        elapsed_seconds = _time.monotonic() - t_start
        log(f"        {progress_label}: done {total_variants:,} variants in {elapsed_seconds:.1f}s  mem={mem()}")
    return result_gpu


def _cupy_asnumpy(cupy: Any, values: Any) -> NDArray:
    asnumpy = getattr(cupy, "asnumpy", None)
    if callable(asnumpy):
        return np.asarray(asnumpy(values))
    return np.asarray(values)


def _sharded_gpu_prefetch_budget_bytes(*, sample_count: int, batch_size: int) -> int:
    """Return per-GPU host prefetch budget for multi-GPU PLINK streaming.

    Sized to keep 2 decoded int8 batches queued per shard so the GPU has a
    next batch ready while the kernel finishes the current one. With the
    fadvise(DONTNEED) drain in _release_host_caches, the kernel page cache
    no longer accumulates, so we can afford more than one in-flight batch
    per shard without OOM-ing the 30 GB AoU box.
    """
    return max(1, int(sample_count) * max(1, int(batch_size)) * 2)


def _gpu_int8_transpose_matmul_sharded(
    *,
    raw_int8: Int8BatchCapable,
    selected_variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    matrix_gpu: Any,
    batch_size: int,
    cupy: Any,
    dtype: Any,
    progress_label: str | None,
    device_ids: Sequence[int],
) -> Any:
    total_variants = int(selected_variant_indices.shape[0])
    splits = _split_contiguous_columns(total_variants, device_ids)
    if len(splits) <= 1:
        return _gpu_int8_transpose_matmul_single_device(
            raw_int8=raw_int8,
            selected_variant_indices=selected_variant_indices,
            means=means,
            scales=scales,
            matrix_gpu=matrix_gpu,
            batch_size=batch_size,
            cupy=cupy,
            dtype=dtype,
            progress_label=progress_label,
            io_worker_limit=None,
            max_prefetch_bytes=None,
            decode_thread_limit=None,
        )
    import time as _time

    t_start = _time.monotonic()
    primary_device_id = _cupy_current_device_id(cupy)
    matrix_host = _cupy_asnumpy(cupy, matrix_gpu)
    cpu_count = max(1, os.cpu_count() or 1)
    per_device_io_workers = max(1, (cpu_count - 2) // max(1, len(splits)))
    per_device_prefetch_bytes = _sharded_gpu_prefetch_budget_bytes(
        sample_count=int(raw_int8.shape[0]),
        batch_size=int(batch_size),
    )
    # No read_semaphore: with fadvise(DONTNEED) on .bed after each batch the
    # OOM rationale for serializing shard reads is gone, and the previous
    # serialization halved decode throughput on the 2x V100 box (both GPUs
    # spent the marginal screen at 0% utilization waiting for the single in-
    # flight PLINK read). Concurrent decodes saturate the disk read pipeline
    # and give each GPU its own decoder.
    if progress_label is not None:
        split_desc = ", ".join(f"gpu{device_id}:{stop - start:,}" for device_id, start, stop in splits)
        log(
            f"        {progress_label}: start multi-GPU streaming {total_variants:,} variants "
            f"across {len(splits)} GPUs ({split_desc}; batch_size={batch_size}; "
            f"prefetch_budget={per_device_prefetch_bytes / 1e6:.0f} MB/device; "
            f"plink_reads=parallel)  mem={mem()}"
        )

    # Resolve the GPU-decode context ONCE (cheap walk) so each shard can
    # use the pread-direct path on its own V100 instead of falling back to
    # the slow CPU-decode + standardize streaming path. Concurrent pread()
    # on the shared open_bed fd is safe (per-call offset, non-overlapping
    # variant slices), and each shard owns its own GPU staging + decode
    # kernel invocation.
    shard_ctx = _resolve_plink_pread_context(raw_int8)

    def _compute_shard(device_id: int, start: int, stop: int) -> tuple[int, int, NDArray]:
        shard_indices = selected_variant_indices[start:stop]
        shard_label = f"{progress_label} gpu{device_id}" if progress_label is not None else None
        with _cupy_device_context(cupy, device_id):
            shard_matrix_gpu = cupy.asarray(matrix_host, dtype=dtype)
            if shard_ctx is not None:
                reader, sample_indices, iid_count, bed_path = shard_ctx
                shard_result_gpu = _gpu_plink_pread_transpose_matmul_direct(
                    reader=reader,
                    sample_indices=sample_indices,
                    iid_count=iid_count,
                    bed_path=bed_path,
                    selected_variant_indices=shard_indices,
                    means=means,
                    scales=scales,
                    matrix_gpu=shard_matrix_gpu,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=dtype,
                    progress_label=shard_label,
                )
            else:
                shard_result_gpu = _gpu_int8_transpose_matmul_single_device(
                    raw_int8=raw_int8,
                    selected_variant_indices=shard_indices,
                    means=means,
                    scales=scales,
                    matrix_gpu=shard_matrix_gpu,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=dtype,
                    progress_label=shard_label,
                    io_worker_limit=per_device_io_workers,
                    max_prefetch_bytes=per_device_prefetch_bytes,
                    decode_thread_limit=max(1, per_device_io_workers),
                    read_semaphore=None,
                )
            _cupy_device_synchronize(cupy, device_id)
            shard_result_host = _cupy_asnumpy(cupy, shard_result_gpu)
        return start, stop, shard_result_host

    pieces: list[tuple[int, int, NDArray]] = []
    with ThreadPoolExecutor(max_workers=len(splits), thread_name_prefix="gpu-int8-transpose") as executor:
        futures = [executor.submit(_compute_shard, device_id, start, stop) for device_id, start, stop in splits]
        for future in futures:
            pieces.append(future.result())
    pieces.sort(key=lambda item: item[0])

    with _cupy_device_context(cupy, primary_device_id):
        result_gpu = cupy.empty((total_variants, matrix_gpu.shape[1]), dtype=dtype)
        for start, stop, shard_result_host in pieces:
            result_gpu[start:stop, :] = cupy.asarray(shard_result_host, dtype=dtype)
    if progress_label is not None:
        elapsed_seconds = _time.monotonic() - t_start
        rate = total_variants / max(elapsed_seconds, 1e-6)
        log(
            f"        {progress_label}: multi-GPU streaming done {total_variants:,} variants "
            f"in {elapsed_seconds:.1f}s rate={rate:,.0f}v/s  mem={mem()}"
        )
    return result_gpu


def _gpu_free_bytes(cupy: Any) -> int:
    """Return free GPU device memory in bytes, or 0 if unavailable."""
    try:
        free, _ = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except _cupy_runtime_error_classes(cupy):
        return 0


def _gpu_total_bytes(cupy: Any) -> int:
    """Return total GPU device memory in bytes, or 0 if unavailable.

    Returns 0 on mocked / partial CuPy stand-ins (used by unit tests that
    don't have a real CUDA runtime) so callers fall back to their static
    defaults instead of crashing.
    """
    try:
        _, total = cupy.cuda.runtime.memGetInfo()
        return int(total)
    except _cupy_runtime_error_classes(cupy):
        return 0


def _gpu_effective_free_bytes(cupy: Any) -> int:
    """Free bytes allocatable by this process: CUDA-free plus CuPy-pool-free blocks.

    ``cupy.cuda.runtime.memGetInfo()`` only sees driver-free memory and does
    NOT count blocks the CuPy memory pool is holding as cached/free. After
    dropping a large ``_cupy_cache``, the pool retains those blocks and
    ``memGetInfo`` can falsely report the GPU as nearly full, causing
    sizing decisions to reject feasible allocations. Sum both.
    """
    free = _gpu_free_bytes(cupy)
    try:
        pool = cupy.get_default_memory_pool()
        pool_free = int(pool.free_bytes())
    except _cupy_runtime_error_classes(cupy):
        pool_free = 0
    return max(0, free + pool_free)


def _gpu_dynamic_standardized_target_batch_bytes(
    cupy: Any,
    *,
    static_target_batch_bytes: int,
    free_fraction: float = GPU_STANDARDIZED_DYNAMIC_FREE_FRACTION,
) -> int:
    free_bytes = _gpu_effective_free_bytes(cupy)
    if free_bytes <= 0:
        return int(static_target_batch_bytes)
    usable_bytes = max(
        free_bytes - GPU_STANDARDIZED_DYNAMIC_RESERVE_BYTES,
        int(free_bytes * 0.10),
    )
    dynamic_target = max(int(usable_bytes * float(free_fraction)), 1)
    return max(1, min(int(static_target_batch_bytes), dynamic_target))


def _is_cupy_out_of_memory(exc: BaseException) -> bool:
    exc_name = exc.__class__.__name__.lower()
    exc_message = str(exc).lower()
    return (
        "outofmemory" in exc_name
        or "out of memory" in exc_message
        or "cuda_error_out_of_memory" in exc_message
    )


def _release_cupy_cached_memory(cupy: Any) -> None:
    """Best-effort release of CuPy pool blocks after a failed allocation."""
    try:
        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
    except _cupy_runtime_error_classes(cupy):
        pass
    try:
        pinned_pool = cupy.get_default_pinned_memory_pool()
        pinned_pool.free_all_blocks()
    except _cupy_runtime_error_classes(cupy):
        pass


def _effective_gpu_standardized_streaming_batch_size(
    *,
    sample_count: int,
    requested_batch_size: int,
    target_batch_bytes: int,
    dtype: Any,
) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive.")
    if target_batch_bytes < 1:
        raise ValueError("target_batch_bytes must be positive.")
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    bytes_per_variant = sample_count * resolved_dtype.itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    return max(1, min(int(requested_batch_size), int(memory_capped_batch_size)))


_GPU_RESERVED_OVERHEAD_BYTES = 1_500_000_000  # 1.5 GB
_GPU_BUDGET_TOTAL_FRACTION_CEILING = 0.90


@dataclass(frozen=True, slots=True)
class _GpuMaterializationMemoryPlan:
    n_rows: int
    n_cols: int
    dtype_name: str
    backend: str
    storage_bytes: int
    metadata_bytes: int
    staging_bytes: int
    result_vector_bytes: int
    safety_margin_bytes: int
    required_bytes: int
    solver_headroom_bytes: int
    budget_bytes: int
    free_bytes: int
    total_bytes: int
    chunk_rows: int

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.budget_bytes


def _bytes_gb(value: int) -> str:
    return f"{int(value) / 1e9:.2f} GB"


def _adaptive_int8_staging_chunk_rows(
    *,
    n_cols: int,
    dtype: Any,
    free_bytes: int,
    cap: int = GPU_INT8_MATMUL_STAGING_ROWS_MAX,
    floor: int = GPU_INT8_MATMUL_STAGING_ROWS,
) -> int:
    """Pick the largest staging chunk that fits the configured free-VRAM slice.

    Returns ``floor`` when free memory is unknown (e.g. mocked CuPy in tests)
    so behavior matches the legacy fixed-default code path. When real free
    memory is available, scales the chunk up to ``cap`` rows as long as the
    fp32-promoted staging slab stays under both the free-fraction budget and
    the absolute ``GPU_INT8_MATMUL_STAGING_MAX_BYTES`` ceiling.
    """
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    compute_itemsize = max(int(resolved_dtype.itemsize), np.dtype(np.float32).itemsize)
    cols = max(int(n_cols), 1)
    bytes_per_row = cols * compute_itemsize
    if free_bytes <= 0 or bytes_per_row <= 0:
        return max(1, int(floor))
    budget_bytes = min(
        int(free_bytes * GPU_INT8_MATMUL_STAGING_FREE_FRACTION),
        int(GPU_INT8_MATMUL_STAGING_MAX_BYTES),
    )
    if budget_bytes < bytes_per_row:
        return max(1, int(floor))
    fits = int(budget_bytes // bytes_per_row)
    # Clamp to [1, cap] using the budget-derived ``fits``. ``floor`` is a soft
    # preference: if ``fits`` falls short of it (very wide cohort, very tight
    # free memory), we still honor ``fits`` rather than blowing past the budget
    # back to ``floor`` and pushing the planner into a forced-streaming path.
    return int(max(1, min(int(cap), fits)))


def _gpu_solver_headroom_bytes(
    *,
    n_rows: int,
    n_cols: int,
    dtype: Any,
    backend: str,
    chunk_rows: int,
    result_vector_count: int,
    safety_margin_bytes: int,
) -> tuple[int, int, int]:
    """Return staging, result-vector, and safety headroom for GPU solves.

    Terms:
    - int8 staging buffer: row-chunked int8-resident matvec promotes only
      ``chunk_rows x n_cols`` values into a reusable fp32/fp64 staging slab.
    - result vectors: sample-space and variant-space vectors that stay live
      during matvec/HVP/TR-Newton iterations.
    - safety margin: CuPy pool fragmentation, CUDA/cuBLAS workspaces, and
      TR-Newton/HVP temporaries not owned by the genotype cache.
    """
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    compute_itemsize = max(int(resolved_dtype.itemsize), np.dtype(np.float32).itemsize)
    safe_chunk_rows = max(1, min(int(chunk_rows), max(int(n_rows), 1)))
    staging_bytes = 0
    if "int8" in backend:
        staging_bytes = safe_chunk_rows * int(n_cols) * compute_itemsize
    vector_count = max(int(result_vector_count), 1)
    result_vector_bytes = vector_count * (int(n_rows) + int(n_cols)) * compute_itemsize
    safety_bytes = max(int(safety_margin_bytes), 0)
    return int(staging_bytes), int(result_vector_bytes), int(safety_bytes)


def _gpu_materialization_budget_bytes(
    cupy: Any,
    *,
    n_rows: int = 0,
    n_cols: int = 0,
    dtype: Any = np.float32,
    backend: str = "unknown",
    chunk_rows: int = GPU_INT8_MATMUL_STAGING_ROWS,
    result_vector_count: int = 2,
    safety_margin_bytes: int = _GPU_RESERVED_OVERHEAD_BYTES,
) -> int:
    """GPU cache bytes available after live-memory and solver headroom.

    Start with ``total_mem * 0.9`` capped by allocatable free memory. Then
    subtract solver headroom: int8 fp32 staging, live result vectors, and a
    1.5 GB safety margin for TR-Newton/HVP temporaries plus CuPy pool
    fragmentation. Cached CuPy pool blocks are released before measuring free
    bytes; live CuPy/JAX allocations remain counted against ``free``.
    """
    total = _gpu_total_bytes(cupy)
    if total <= 0:
        return 0
    try:
        cupy.cuda.Device().synchronize()
    except (AttributeError, OSError, RuntimeError):
        pass
    _release_cupy_cached_memory(cupy)
    free = _gpu_effective_free_bytes(cupy)
    capped_available = int(total * _GPU_BUDGET_TOTAL_FRACTION_CEILING)
    if free > 0:
        capped_available = min(capped_available, int(free))
    staging_bytes, result_vector_bytes, safety_bytes = _gpu_solver_headroom_bytes(
        n_rows=max(int(n_rows), 0),
        n_cols=max(int(n_cols), 0),
        dtype=dtype,
        backend=backend,
        chunk_rows=chunk_rows,
        result_vector_count=result_vector_count,
        safety_margin_bytes=safety_margin_bytes,
    )
    return max(capped_available - staging_bytes - result_vector_bytes - safety_bytes, 0)


def _call_gpu_materialization_budget_bytes(cupy: Any, **kwargs: Any) -> int:
    try:
        return _gpu_materialization_budget_bytes(cupy, **kwargs)
    except TypeError:
        return _gpu_materialization_budget_bytes(cupy)


def _estimate_gpu_materialization_memory_plan(
    *,
    n_rows: int,
    n_cols: int,
    dtype: Any,
    backend: str,
    cupy: Any,
    metadata_bytes: int = 0,
    chunk_rows: int = GPU_INT8_MATMUL_STAGING_ROWS,
    result_vector_count: int = 2,
    safety_margin_bytes: int = _GPU_RESERVED_OVERHEAD_BYTES,
) -> _GpuMaterializationMemoryPlan:
    resolved_dtype = _cupy_dtype_to_numpy_dtype(dtype)
    rows = max(int(n_rows), 0)
    cols = max(int(n_cols), 0)
    storage_bytes = rows * cols * int(resolved_dtype.itemsize)
    # Resolve the staging chunk adaptively: scale up to fit free VRAM when the
    # caller passed the default, but respect any explicit override (callers
    # that need to plan with a specific chunk size — e.g. matmul code paths
    # that have already allocated the slab — pass their own value).
    effective_chunk_rows = int(chunk_rows)
    if "int8" in backend and effective_chunk_rows <= GPU_INT8_MATMUL_STAGING_ROWS:
        try:
            free_bytes_for_chunk = _gpu_effective_free_bytes(cupy)
        except (AttributeError, OSError, RuntimeError):
            free_bytes_for_chunk = 0
        if free_bytes_for_chunk > 0:
            effective_chunk_rows = _adaptive_int8_staging_chunk_rows(
                n_cols=cols,
                dtype=dtype,
                free_bytes=free_bytes_for_chunk,
            )
    staging_bytes, result_vector_bytes, safety_bytes = _gpu_solver_headroom_bytes(
        n_rows=rows,
        n_cols=cols,
        dtype=dtype,
        backend=backend,
        chunk_rows=effective_chunk_rows,
        result_vector_count=result_vector_count,
        safety_margin_bytes=safety_margin_bytes,
    )
    budget_bytes = _call_gpu_materialization_budget_bytes(
        cupy,
        n_rows=rows,
        n_cols=cols,
        dtype=dtype,
        backend=backend,
        chunk_rows=effective_chunk_rows,
        result_vector_count=result_vector_count,
        safety_margin_bytes=safety_margin_bytes,
    )
    return _GpuMaterializationMemoryPlan(
        n_rows=rows,
        n_cols=cols,
        dtype_name=resolved_dtype.name,
        backend=backend,
        storage_bytes=int(storage_bytes),
        metadata_bytes=int(metadata_bytes),
        staging_bytes=int(staging_bytes),
        result_vector_bytes=int(result_vector_bytes),
        safety_margin_bytes=int(safety_bytes),
        required_bytes=int(storage_bytes + metadata_bytes),
        solver_headroom_bytes=int(staging_bytes + result_vector_bytes + safety_bytes),
        budget_bytes=int(budget_bytes),
        free_bytes=_gpu_effective_free_bytes(cupy),
        total_bytes=_gpu_total_bytes(cupy),
        chunk_rows=max(1, min(int(effective_chunk_rows), max(rows, 1))),
    )


def _log_gpu_materialization_memory_plan(plan: _GpuMaterializationMemoryPlan) -> None:
    log(
        "    GPU materialization plan: "
        + f"backend={plan.backend} dtype={plan.dtype_name} shape={plan.n_rows}x{plan.n_cols} "
        + f"storage={_bytes_gb(plan.storage_bytes)} metadata={_bytes_gb(plan.metadata_bytes)} "
        + f"staging={_bytes_gb(plan.staging_bytes)}({plan.chunk_rows} rows) "
        + f"result_vectors={_bytes_gb(plan.result_vector_bytes)} "
        + f"safety={_bytes_gb(plan.safety_margin_bytes)} "
        + f"headroom={_bytes_gb(plan.solver_headroom_bytes)} "
        + f"budget={_bytes_gb(plan.budget_bytes)} free={_bytes_gb(plan.free_bytes)} "
        + f"total={_bytes_gb(plan.total_bytes)}  mem={mem()}"
    )


def _warn_gpu_materialization_unavailable(reason: str, plan: _GpuMaterializationMemoryPlan) -> None:
    log(
        "    WARNING: GPU materialization unavailable; streaming will be used. "
        + f"reason={reason}; backend={plan.backend} dtype={plan.dtype_name} "
        + f"required={_bytes_gb(plan.required_bytes)} budget={_bytes_gb(plan.budget_bytes)} "
        + f"storage={_bytes_gb(plan.storage_bytes)} metadata={_bytes_gb(plan.metadata_bytes)} "
        + f"headroom={_bytes_gb(plan.solver_headroom_bytes)} "
        + f"staging={_bytes_gb(plan.staging_bytes)} result_vectors={_bytes_gb(plan.result_vector_bytes)} "
        + f"safety={_bytes_gb(plan.safety_margin_bytes)} free={_bytes_gb(plan.free_bytes)} "
        + f"total={_bytes_gb(plan.total_bytes)}  mem={mem()}"
    )


@dataclass(slots=True)
class _CupyInt8StandardizedCache:
    raw_values: Any
    means: Any
    scales: Any
    cupy: Any = field(repr=False)
    # When set, this cache is a view over ``raw_values``: the i-th logical
    # column maps to ``raw_values[:, column_indices[i]]``. ``means`` / ``scales``
    # are always pre-subset to logical column order, so they are indexed
    # directly with logical positions.
    column_indices: Any | None = None

    @property
    def shape(self) -> tuple[int, int]:
        rows = int(self.raw_values.shape[0])
        if self.column_indices is None:
            return rows, int(self.raw_values.shape[1])
        return rows, int(self.column_indices.shape[0])

    @property
    def nbytes(self) -> int:
        # Report the *logical* footprint so callers (budget logs, materialization
        # accounting) see the view's effective size rather than the shared parent
        # buffer. Root caches (column_indices is None) report the true allocation.
        means_bytes = int(self.means.nbytes)
        scales_bytes = int(self.scales.nbytes)
        if self.column_indices is None:
            return int(self.raw_values.nbytes) + means_bytes + scales_bytes
        sample_count = int(self.raw_values.shape[0])
        view_columns = int(self.column_indices.shape[0])
        itemsize = int(self.raw_values.dtype.itemsize)
        return sample_count * view_columns * itemsize + means_bytes + scales_bytes + int(self.column_indices.nbytes)

    def subset(self, local_variant_indices: NDArray) -> _CupyInt8StandardizedCache:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if _local_indices_select_all(resolved_local_indices, self.shape[1]):
            return self
        cp = self.cupy
        local_slice = _local_indices_as_slice(resolved_local_indices)
        if local_slice is not None:
            # Contiguous selection: subset means/scales by slice (cheap) and
            # either keep a zero-copy slice view of raw_values (root cache) or
            # slice the parent's column-index array (view-of-view). Both paths
            # avoid materializing a fresh genotype copy on the device.
            means = self.means[local_slice]
            scales = self.scales[local_slice]
            if self.column_indices is None:
                return _CupyInt8StandardizedCache(
                    raw_values=self.raw_values[:, local_slice],
                    means=means,
                    scales=scales,
                    cupy=cp,
                )
            return _CupyInt8StandardizedCache(
                raw_values=self.raw_values,
                means=means,
                scales=scales,
                cupy=cp,
                column_indices=self.column_indices[local_slice],
            )
        # Non-contiguous fancy selection: defer the raw_values gather. Keep
        # the parent buffer shared and store the resolved column ids on the
        # device, turning ``subset`` from O(samples x selected) device bytes
        # into O(selected) device bytes; critical when ``selected`` covers
        # most of the cache and would otherwise OOM by duplicating it.
        # ``standardized_columns`` gathers each working batch on demand.
        means = self.means[resolved_local_indices]
        scales = self.scales[resolved_local_indices]
        device_indices = cp.asarray(resolved_local_indices)
        composed_indices = (
            device_indices if self.column_indices is None else self.column_indices[device_indices]
        )
        return _CupyInt8StandardizedCache(
            raw_values=self.raw_values,
            means=means,
            scales=scales,
            cupy=cp,
            column_indices=composed_indices,
        )

    def _resolve_raw_selector(self, sel: Any) -> Any:
        """Map a logical column selector onto the underlying ``raw_values``."""
        if self.column_indices is None:
            return sel
        return self.column_indices[sel]

    def standardized_columns(
        self,
        local_variant_indices: NDArray | slice,
        *,
        dtype: Any = None,
    ) -> Any:
        cp = self.cupy
        resolved_dtype = _cupy_compute_dtype(cp) if dtype is None else dtype
        raw_chunk = self.raw_values[:, self._resolve_raw_selector(local_variant_indices)]
        means = self.means[local_variant_indices].astype(resolved_dtype, copy=False)
        scales = self.scales[local_variant_indices].astype(resolved_dtype, copy=False)
        standardized = _standardize_int8_cupy(
            raw_chunk,
            means,
            scales,
            cp,
            dtype=resolved_dtype,
        )
        if hasattr(cp, "asarray"):
            return cp.asarray(standardized, dtype=resolved_dtype)
        standardized_np = np.asarray(standardized)
        return standardized_np

    def __array__(self, dtype: np.dtype[Any] | type | None = None) -> NDArray:
        standardized = self.standardized_columns(slice(None), dtype=np.float32)
        host = standardized.get() if hasattr(standardized, "get") else standardized
        return np.asarray(host, dtype=np.float32 if dtype is None else dtype)


@dataclass(frozen=True, slots=True)
class _CupyDeviceCacheShard:
    device_id: int
    column_start: int
    cache: Any

    @property
    def column_stop(self) -> int:
        return self.column_start + _cupy_cache_shape(self.cache)[1]


@dataclass(frozen=True, slots=True)
class _CupyShardedStandardizedCache:
    shards: tuple[_CupyDeviceCacheShard, ...]
    cupy: Any = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        if not self.shards:
            return 0, 0
        rows = _cupy_cache_shape(self.shards[0].cache)[0]
        columns = sum(_cupy_cache_shape(shard.cache)[1] for shard in self.shards)
        return rows, int(columns)

    @property
    def nbytes(self) -> int:
        return sum(_cupy_cache_nbytes(shard.cache) for shard in self.shards)

    def subset(self, local_variant_indices: NDArray) -> _CupyShardedStandardizedCache:
        resolved = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved.size > 1 and np.any(np.diff(resolved) < 0):
            raise ValueError("multi-GPU genotype cache subsets must be sorted by column.")
        if _local_indices_select_all(resolved, self.shape[1]):
            return self
        next_shards: list[_CupyDeviceCacheShard] = []
        next_column_start = 0
        for shard in self.shards:
            shard_start = int(shard.column_start)
            shard_stop = int(shard.column_stop)
            mask = (resolved >= shard_start) & (resolved < shard_stop)
            if not bool(np.any(mask)):
                continue
            shard_indices = (resolved[mask] - shard_start).astype(np.int32, copy=False)
            with _cupy_device_context(self.cupy, shard.device_id):
                shard_cache = _cupy_cache_subset_columns(shard.cache, shard_indices)
            next_shards.append(
                _CupyDeviceCacheShard(
                    device_id=shard.device_id,
                    column_start=next_column_start,
                    cache=shard_cache,
                )
            )
            next_column_start += int(shard_indices.shape[0])
        return _CupyShardedStandardizedCache(tuple(next_shards), cupy=self.cupy)

    def standardized_columns(
        self,
        local_variant_indices: NDArray | slice,
        *,
        dtype: Any = None,
        device_id: int = 0,
    ) -> Any:
        if isinstance(local_variant_indices, slice):
            selected = np.arange(self.shape[1], dtype=np.int32)[local_variant_indices]
        else:
            selected = np.asarray(local_variant_indices, dtype=np.int32)
        subset = self.subset(selected)
        return subset.to_device_array(dtype=dtype, device_id=device_id)

    def to_device_array(self, *, dtype: Any = None, device_id: int = 0) -> Any:
        cp = self.cupy
        if not self.shards:
            with _cupy_device_context(cp, device_id):
                resolved_dtype = _cupy_compute_dtype(cp) if dtype is None else dtype
                return cp.empty((0, 0), dtype=resolved_dtype)
        resolved_dtype = _cupy_compute_dtype(cp) if dtype is None else dtype
        pieces: list[Any] = []
        for shard in self.shards:
            with _cupy_device_context(cp, shard.device_id):
                piece = _cupy_cache_standardized_columns(
                    shard.cache,
                    slice(None),
                    cupy=cp,
                    dtype=resolved_dtype,
                )
            with _cupy_device_context(cp, device_id):
                pieces.append(cp.asarray(piece, dtype=resolved_dtype))
        with _cupy_device_context(cp, device_id):
            concatenate = getattr(cp, "concatenate", np.concatenate)
            return concatenate(pieces, axis=1)

    def __array__(self, dtype: np.dtype[Any] | type | None = None) -> NDArray:
        array = self.to_device_array(dtype=np.float32, device_id=0)
        host = array.get() if hasattr(array, "get") else array
        return np.asarray(host, dtype=np.float32 if dtype is None else dtype)


def _cupy_cache_is_int8_standardized(cache: Any | None) -> TypeGuard[_CupyInt8StandardizedCache]:
    return isinstance(cache, _CupyInt8StandardizedCache)


def _cupy_cache_is_sharded(cache: Any | None) -> TypeGuard[_CupyShardedStandardizedCache]:
    return isinstance(cache, _CupyShardedStandardizedCache)


def _cupy_cache_shape(cache: Any) -> tuple[int, int]:
    if _cupy_cache_is_sharded(cache):
        return cache.shape
    if _cupy_cache_is_int8_standardized(cache):
        return cache.shape
    return int(cache.shape[0]), int(cache.shape[1])


def _cupy_cache_nbytes(cache: Any) -> int:
    if _cupy_cache_is_sharded(cache):
        return cache.nbytes
    if _cupy_cache_is_int8_standardized(cache):
        return cache.nbytes
    return int(cache.nbytes)


def _cupy_cache_subset_columns(cache: Any, local_variant_indices: NDArray) -> Any:
    resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
    if _cupy_cache_is_sharded(cache):
        return cache.subset(resolved_local_indices)
    if _cupy_cache_is_int8_standardized(cache):
        return cache.subset(resolved_local_indices)
    if _local_indices_select_all(resolved_local_indices, int(cache.shape[1])):
        return cache
    local_slice = _local_indices_as_slice(resolved_local_indices)
    if local_slice is not None:
        return cache[:, local_slice]
    return cache[:, resolved_local_indices]


def _cupy_cache_standardized_columns(
    cache: Any,
    local_variant_indices: NDArray | slice,
    *,
    cupy: Any,
    dtype: Any = None,
) -> Any:
    if _cupy_cache_is_sharded(cache):
        return cache.standardized_columns(
            local_variant_indices,
            dtype=dtype,
            device_id=_cupy_current_device_id(cupy),
        )
    if _cupy_cache_is_int8_standardized(cache):
        return cache.standardized_columns(local_variant_indices, dtype=dtype)
    resolved_dtype = _cupy_compute_dtype(cupy) if dtype is None else dtype
    return cache[:, local_variant_indices].astype(resolved_dtype, copy=False)


def _cupy_cache_is_fp16_resident(cache: Any | None) -> bool:
    if cache is None or _cupy_cache_is_int8_standardized(cache) or _cupy_cache_is_sharded(cache):
        return False
    try:
        return np.dtype(cache.dtype) == np.dtype(np.float16)
    except (AttributeError, TypeError):
        return False


def _dense_array_cache_available(cache: Any | None) -> bool:
    return (
        cache is not None
        and not _cupy_cache_is_int8_standardized(cache)
        and not _cupy_cache_is_sharded(cache)
        and not _cupy_cache_is_fp16_resident(cache)
    )


def _gpu_int8_cache_variant_matmul(
    cache: _CupyInt8StandardizedCache,
    matrix_gpu: Any,
    *,
    local_variant_indices: NDArray | None,
    cupy: Any,
    dtype: Any,
) -> Any:
    selected_count = int(matrix_gpu.shape[0])
    if selected_count == 0:
        return cupy.zeros((cache.shape[0], matrix_gpu.shape[1]), dtype=dtype)
    if local_variant_indices is None:
        selector: Any = slice(None)
        means = cache.means.astype(dtype, copy=False)
        scales = cache.scales.astype(dtype, copy=False)
    else:
        selector = np.asarray(local_variant_indices, dtype=np.int32)
        means = cache.means[selector].astype(dtype, copy=False)
        scales = cache.scales[selector].astype(dtype, copy=False)
    raw_selector = cache._resolve_raw_selector(selector)
    chunk_rows = max(1, min(GPU_INT8_MATMUL_STAGING_ROWS, int(cache.shape[0])))
    empty_fn = getattr(cupy, "empty", np.empty)
    staging = empty_fn((chunk_rows, selected_count), dtype=dtype)
    result_gpu = empty_fn((cache.shape[0], matrix_gpu.shape[1]), dtype=dtype)
    for row_start in range(0, cache.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, cache.shape[0])
        row_slice = slice(row_start, row_stop)
        active_rows = row_stop - row_start
        staging_chunk = staging[:active_rows, :]
        raw_chunk = cache.raw_values[row_slice, raw_selector]
        _standardize_int8_cupy_into(raw_chunk, means, scales, staging_chunk, cupy, dtype=dtype)
        result_gpu[row_slice, :] = staging_chunk @ matrix_gpu
    return result_gpu


def _gpu_int8_cache_transpose_matmul(
    cache: _CupyInt8StandardizedCache,
    matrix_gpu: Any,
    *,
    cupy: Any,
    dtype: Any,
) -> Any:
    variant_count = int(cache.shape[1])
    if variant_count == 0:
        return cupy.zeros((0, matrix_gpu.shape[1]), dtype=dtype)
    means = cache.means.astype(dtype, copy=False)
    scales = cache.scales.astype(dtype, copy=False)
    raw_selector = cache._resolve_raw_selector(slice(None))
    chunk_rows = max(1, min(GPU_INT8_MATMUL_STAGING_ROWS, int(cache.shape[0])))
    empty_fn = getattr(cupy, "empty", np.empty)
    zeros_fn = getattr(cupy, "zeros", np.zeros)
    staging = empty_fn((chunk_rows, variant_count), dtype=dtype)
    result_gpu = zeros_fn((variant_count, matrix_gpu.shape[1]), dtype=dtype)
    for row_start in range(0, cache.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, cache.shape[0])
        row_slice = slice(row_start, row_stop)
        active_rows = row_stop - row_start
        staging_chunk = staging[:active_rows, :]
        raw_chunk = cache.raw_values[row_slice, raw_selector]
        _standardize_int8_cupy_into(raw_chunk, means, scales, staging_chunk, cupy, dtype=dtype)
        result_gpu += staging_chunk.T @ matrix_gpu[row_slice, :]
    return result_gpu


def _gpu_fp16_cache_variant_matmul(
    cache: Any,
    matrix_gpu: Any,
    *,
    local_variant_indices: NDArray | None,
    cupy: Any,
    dtype: Any,
) -> Any:
    fp16_dtype = getattr(cupy, "float16", np.float16)
    rhs = matrix_gpu.astype(fp16_dtype, copy=False)
    lhs = cache if local_variant_indices is None else cache[:, np.asarray(local_variant_indices, dtype=np.int32)]
    return (lhs @ rhs).astype(dtype, copy=False)


def _gpu_fp16_cache_transpose_matmul(cache: Any, matrix_gpu: Any, *, cupy: Any, dtype: Any) -> Any:
    fp16_dtype = getattr(cupy, "float16", np.float16)
    rhs = matrix_gpu.astype(fp16_dtype, copy=False)
    return (cache.T @ rhs).astype(dtype, copy=False)


def _gpu_single_cache_variant_matmul(
    cache: Any,
    matrix_gpu: Any,
    *,
    local_variant_indices: NDArray | None,
    cupy: Any,
    dtype: Any,
) -> Any:
    if _cupy_cache_is_int8_standardized(cache):
        return _gpu_int8_cache_variant_matmul(
            cache,
            matrix_gpu,
            local_variant_indices=local_variant_indices,
            cupy=cupy,
            dtype=dtype,
        )
    if _cupy_cache_is_fp16_resident(cache):
        return _gpu_fp16_cache_variant_matmul(
            cache,
            matrix_gpu,
            local_variant_indices=local_variant_indices,
            cupy=cupy,
            dtype=dtype,
        )
    if local_variant_indices is None:
        return cache.astype(dtype, copy=False) @ matrix_gpu
    return cache[:, np.asarray(local_variant_indices, dtype=np.int32)].astype(dtype, copy=False) @ matrix_gpu


def _gpu_single_cache_transpose_matmul(cache: Any, matrix_gpu: Any, *, cupy: Any, dtype: Any) -> Any:
    if _cupy_cache_is_int8_standardized(cache):
        return _gpu_int8_cache_transpose_matmul(cache, matrix_gpu, cupy=cupy, dtype=dtype)
    if _cupy_cache_is_fp16_resident(cache):
        return _gpu_fp16_cache_transpose_matmul(cache, matrix_gpu, cupy=cupy, dtype=dtype)
    return cache.astype(dtype, copy=False).T @ matrix_gpu


def _sharded_variant_groups(
    cache: _CupyShardedStandardizedCache,
    local_variant_indices: NDArray | None,
) -> tuple[tuple[_CupyDeviceCacheShard, NDArray | None, NDArray | None], ...]:
    if local_variant_indices is None:
        return tuple((shard, None, None) for shard in cache.shards)
    resolved = np.asarray(local_variant_indices, dtype=np.int32)
    groups: list[tuple[_CupyDeviceCacheShard, NDArray | None, NDArray | None]] = []
    for shard in cache.shards:
        shard_start = int(shard.column_start)
        shard_stop = int(shard.column_stop)
        operand_positions = np.flatnonzero((resolved >= shard_start) & (resolved < shard_stop)).astype(np.int32)
        if operand_positions.size == 0:
            continue
        shard_local_indices = (resolved[operand_positions] - shard_start).astype(np.int32, copy=False)
        groups.append((shard, operand_positions, shard_local_indices))
    return tuple(groups)


def _gpu_sharded_cache_variant_matmul(
    cache: _CupyShardedStandardizedCache,
    matrix_gpu: Any,
    *,
    local_variant_indices: NDArray | None,
    cupy: Any,
    dtype: Any,
) -> Any:
    groups = _sharded_variant_groups(cache, local_variant_indices)
    if not groups:
        with _cupy_device_context(cupy, 0):
            return cupy.zeros((cache.shape[0], matrix_gpu.shape[1]), dtype=dtype)

    def compute(group: tuple[_CupyDeviceCacheShard, NDArray | None, NDArray | None]) -> Any:
        shard, operand_positions, shard_local_indices = group
        with _cupy_device_context(cupy, shard.device_id):
            if operand_positions is None:
                shard_matrix = cupy.asarray(
                    matrix_gpu[shard.column_start:shard.column_stop, :],
                    dtype=dtype,
                )
            else:
                shard_matrix = cupy.asarray(matrix_gpu[operand_positions, :], dtype=dtype)
            result = _gpu_single_cache_variant_matmul(
                shard.cache,
                shard_matrix,
                local_variant_indices=shard_local_indices,
                cupy=cupy,
                dtype=dtype,
            )
            _cupy_device_synchronize(cupy, shard.device_id)
            return result

    primary_device_id = int(groups[0][0].device_id)
    with _cupy_device_context(cupy, primary_device_id):
        result_gpu = cupy.zeros((cache.shape[0], matrix_gpu.shape[1]), dtype=dtype)
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(compute, group) for group in groups]
        for future in futures:
            partial = future.result()
            with _cupy_device_context(cupy, primary_device_id):
                result_gpu += cupy.asarray(partial, dtype=dtype)
    return result_gpu


def _gpu_sharded_cache_transpose_matmul(
    cache: _CupyShardedStandardizedCache,
    matrix_gpu: Any,
    *,
    cupy: Any,
    dtype: Any,
) -> Any:
    if not cache.shards:
        with _cupy_device_context(cupy, 0):
            return cupy.zeros((0, matrix_gpu.shape[1]), dtype=dtype)

    def compute(shard: _CupyDeviceCacheShard) -> Any:
        with _cupy_device_context(cupy, shard.device_id):
            shard_matrix = cupy.asarray(matrix_gpu, dtype=dtype)
            result = _gpu_single_cache_transpose_matmul(shard.cache, shard_matrix, cupy=cupy, dtype=dtype)
            _cupy_device_synchronize(cupy, shard.device_id)
            return result

    primary_device_id = int(cache.shards[0].device_id)
    with ThreadPoolExecutor(max_workers=len(cache.shards)) as executor:
        pieces = [future.result() for future in (executor.submit(compute, shard) for shard in cache.shards)]
    with _cupy_device_context(cupy, primary_device_id):
        concatenate = getattr(cupy, "concatenate", np.concatenate)
        return concatenate([cupy.asarray(piece, dtype=dtype) for piece in pieces], axis=0)


def _iter_cupy_cache_standardized_batches(
    cache: Any,
    *,
    sample_count: int,
    batch_size: int,
    cupy: Any,
    dtype: Any = None,
) -> Iterator[tuple[slice, Any]]:
    variant_count = _cupy_cache_shape(cache)[1]
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    static_target_batch_bytes = (
        LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        if _cupy_cache_is_int8_standardized(cache)
        else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
    )
    target_batch_bytes = _gpu_dynamic_standardized_target_batch_bytes(
        cupy,
        static_target_batch_bytes=static_target_batch_bytes,
    )
    bytes_per_variant = sample_count * resolved_dtype.itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    safe_batch_size = max(1, min(max(int(batch_size), 1), memory_capped_batch_size))
    for start_index in range(0, variant_count, safe_batch_size):
        stop_index = min(start_index + safe_batch_size, variant_count)
        batch_slice = slice(start_index, stop_index)
        yield batch_slice, _cupy_cache_standardized_columns(
            cache,
            batch_slice,
            cupy=cupy,
            dtype=dtype,
        )


def _iter_selected_cupy_cache_standardized_batches(
    cache: Any,
    local_variant_indices: NDArray,
    *,
    sample_count: int,
    batch_size: int,
    cupy: Any,
    dtype: Any = None,
) -> Iterator[tuple[slice, Any]]:
    selected_variant_count = int(local_variant_indices.shape[0])
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    static_target_batch_bytes = (
        LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        if _cupy_cache_is_int8_standardized(cache)
        else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
    )
    target_batch_bytes = _gpu_dynamic_standardized_target_batch_bytes(
        cupy,
        static_target_batch_bytes=static_target_batch_bytes,
    )
    bytes_per_variant = sample_count * resolved_dtype.itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    safe_batch_size = max(1, min(max(int(batch_size), 1), memory_capped_batch_size))
    for start_index in range(0, selected_variant_count, safe_batch_size):
        stop_index = min(start_index + safe_batch_size, selected_variant_count)
        operand_slice = slice(start_index, stop_index)
        selected_indices = local_variant_indices[operand_slice]
        maybe_slice = _local_indices_as_slice(selected_indices)
        local_selection: NDArray | slice = maybe_slice if maybe_slice is not None else selected_indices
        yield operand_slice, _cupy_cache_standardized_columns(
            cache,
            local_selection,
            cupy=cupy,
            dtype=dtype,
        )


def _normalize_numpy_vector_operand(
    operand: NDArray | JaxArray,
    *,
    expected_length: int,
    shape_error: str,
    finite_name: str,
) -> NDArray:
    vector = np.asarray(operand, dtype=gpu_compute_numpy_dtype()).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(shape_error)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{finite_name} must contain only finite values.")
    return vector


def _normalize_numpy_matrix_operand(
    operand: NDArray | JaxArray,
    *,
    expected_rows: int,
    shape_error: str,
    finite_name: str,
) -> NDArray:
    matrix = np.asarray(operand, dtype=gpu_compute_numpy_dtype())
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(shape_error)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{finite_name} must contain only finite values.")
    return matrix


def _normalize_jax_vector_operand(
    operand: NDArray | JaxArray,
    *,
    expected_length: int,
    shape_error: str,
    finite_name: str,
) -> jax.Array:
    vector = jnp.ravel(jnp.asarray(operand, dtype=gpu_compute_jax_dtype()))
    if vector.shape[0] != expected_length:
        raise ValueError(shape_error)
    if not isinstance(vector, jax_core.Tracer):
        if not np.all(np.isfinite(np.asarray(vector))):
            raise ValueError(f"{finite_name} must contain only finite values.")
    return vector


def _normalize_jax_matrix_operand(
    operand: NDArray | JaxArray,
    *,
    expected_rows: int,
    shape_error: str,
    finite_name: str,
) -> jax.Array:
    matrix = jnp.asarray(operand, dtype=gpu_compute_jax_dtype())
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(shape_error)
    if not isinstance(matrix, jax_core.Tracer):
        if not np.all(np.isfinite(np.asarray(matrix))):
            raise ValueError(f"{finite_name} must contain only finite values.")
    return matrix


def _active_vector_local_indices(vector: NDArray) -> NDArray:
    return np.flatnonzero(vector != 0).astype(np.int32, copy=False)


def _active_matrix_row_local_indices(matrix: NDArray) -> NDArray:
    return np.flatnonzero(np.any(matrix != 0, axis=1)).astype(np.int32, copy=False)


def _local_indices_select_all(local_variant_indices: NDArray, variant_count: int) -> bool:
    if local_variant_indices.shape[0] != variant_count:
        return False
    if variant_count == 0:
        return True
    return bool(
        local_variant_indices[0] == 0
        and local_variant_indices[-1] == variant_count - 1
        and np.all(local_variant_indices == np.arange(variant_count, dtype=local_variant_indices.dtype))
    )


def _local_indices_as_slice(local_variant_indices: NDArray) -> slice | None:
    if local_variant_indices.size == 0:
        return slice(0, 0)
    start = int(local_variant_indices[0])
    stop = int(local_variant_indices[-1]) + 1
    if stop < start:
        return None
    if stop - start != int(local_variant_indices.size):
        return None
    if bool(np.all(local_variant_indices == np.arange(start, stop, dtype=local_variant_indices.dtype))):
        return slice(start, stop)
    return None


def _selected_or_all_local_indices(local_variant_indices: NDArray, variant_count: int) -> NDArray | None:
    return None if _local_indices_select_all(local_variant_indices, variant_count) else local_variant_indices


@dataclass(slots=True)
class _SparseCarrierBackend:
    sample_count: int
    means: NDArray
    scales: NDArray
    variant_ptr: NDArray
    variant_sample_indices: NDArray
    variant_dosages: NDArray
    missing_variant_ptr: NDArray
    missing_variant_sample_indices: NDArray
    sample_ptr: NDArray
    sample_variant_indices: NDArray
    sample_dosages: NDArray
    sample_missing_ptr: NDArray
    sample_missing_variant_indices: NDArray

    @property
    def shape(self) -> tuple[int, int]:
        return self.sample_count, int(self.means.shape[0])

    def materialize_columns(self, local_variant_indices: NDArray) -> NDArray:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        output = np.empty((self.sample_count, resolved_local_indices.shape[0]), dtype=np.float32)
        for output_column, local_variant_index in enumerate(resolved_local_indices.tolist()):
            baseline = -float(self.means[local_variant_index] / self.scales[local_variant_index])
            column = np.full(self.sample_count, baseline, dtype=np.float32)
            start = int(self.variant_ptr[local_variant_index])
            stop = int(self.variant_ptr[local_variant_index + 1])
            if stop > start:
                sample_indices = self.variant_sample_indices[start:stop]
                dosages = self.variant_dosages[start:stop]
                column[sample_indices] += dosages / self.scales[local_variant_index]
            missing_start = int(self.missing_variant_ptr[local_variant_index])
            missing_stop = int(self.missing_variant_ptr[local_variant_index + 1])
            if missing_stop > missing_start:
                column[self.missing_variant_sample_indices[missing_start:missing_stop]] = 0.0
            output[:, output_column] = column
        return output

    def matvec(self, coefficients: NDArray) -> NDArray:
        coefficient_array = np.asarray(coefficients, dtype=gpu_compute_numpy_dtype()).reshape(-1)
        if coefficient_array.shape[0] != self.shape[1]:
            raise ValueError("coefficient vector must match sparse variant count.")
        if not np.any(coefficient_array):
            return np.zeros(self.sample_count, dtype=coefficient_array.dtype)
        scaled_coefficients = coefficient_array / np.asarray(self.scales, dtype=coefficient_array.dtype)
        result = np.full(
            self.sample_count,
            -float(np.dot(np.asarray(self.means, dtype=coefficient_array.dtype), scaled_coefficients)),
            dtype=coefficient_array.dtype,
        )
        if self.sample_variant_indices.size > 0:
            carrier_weights = self.sample_dosages.astype(coefficient_array.dtype, copy=False) * scaled_coefficients[self.sample_variant_indices]
            nonempty_samples = self.sample_ptr[1:] > self.sample_ptr[:-1]
            if np.any(nonempty_samples):
                starts = self.sample_ptr[:-1][nonempty_samples]
                result[nonempty_samples] += np.add.reduceat(carrier_weights, starts)
        if self.sample_missing_variant_indices.size > 0:
            missing_weights = np.asarray(
                self.means[self.sample_missing_variant_indices] * scaled_coefficients[self.sample_missing_variant_indices],
                dtype=coefficient_array.dtype,
            )
            nonempty_missing = self.sample_missing_ptr[1:] > self.sample_missing_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.sample_missing_ptr[:-1][nonempty_missing]
                result[nonempty_missing] += np.add.reduceat(missing_weights, starts)
        return result

    def matmat(self, matrix: NDArray) -> NDArray:
        matrix_array = np.asarray(matrix, dtype=gpu_compute_numpy_dtype())
        if matrix_array.ndim != 2 or matrix_array.shape[0] != self.shape[1]:
            raise ValueError("variant matrix must match sparse variant count.")
        if not np.any(matrix_array):
            return np.zeros((self.sample_count, matrix_array.shape[1]), dtype=matrix_array.dtype)
        scaled_matrix = matrix_array / np.asarray(self.scales, dtype=matrix_array.dtype)[:, None]
        output = np.broadcast_to(
            -np.sum(np.asarray(self.means, dtype=matrix_array.dtype)[:, None] * scaled_matrix, axis=0, dtype=matrix_array.dtype),
            (self.sample_count, matrix_array.shape[1]),
        ).copy()
        if self.sample_variant_indices.size > 0:
            carrier_weights = self.sample_dosages.astype(matrix_array.dtype, copy=False)[:, None] * scaled_matrix[self.sample_variant_indices, :]
            nonempty_samples = self.sample_ptr[1:] > self.sample_ptr[:-1]
            if np.any(nonempty_samples):
                starts = self.sample_ptr[:-1][nonempty_samples]
                output[nonempty_samples, :] += np.add.reduceat(carrier_weights, starts, axis=0)
        if self.sample_missing_variant_indices.size > 0:
            missing_weights = np.asarray(
                self.means[self.sample_missing_variant_indices, None] * scaled_matrix[self.sample_missing_variant_indices, :],
                dtype=matrix_array.dtype,
            )
            nonempty_missing = self.sample_missing_ptr[1:] > self.sample_missing_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.sample_missing_ptr[:-1][nonempty_missing]
                output[nonempty_missing, :] += np.add.reduceat(missing_weights, starts, axis=0)
        return output

    def transpose_matvec(self, vector: NDArray) -> NDArray:
        vector_array = np.asarray(vector, dtype=gpu_compute_numpy_dtype()).reshape(-1)
        if vector_array.shape[0] != self.sample_count:
            raise ValueError("sample vector must match sparse sample count.")
        if not np.any(vector_array):
            return np.zeros(self.shape[1], dtype=vector_array.dtype)
        output = np.zeros(self.shape[1], dtype=vector_array.dtype)
        global_sum = float(np.sum(vector_array, dtype=vector_array.dtype))
        if self.variant_sample_indices.size > 0:
            carrier_weights = self.variant_dosages.astype(vector_array.dtype, copy=False) * vector_array[self.variant_sample_indices]
            nonempty_variants = self.variant_ptr[1:] > self.variant_ptr[:-1]
            if np.any(nonempty_variants):
                starts = self.variant_ptr[:-1][nonempty_variants]
                output[nonempty_variants] += np.add.reduceat(carrier_weights, starts)
        observed_sums = np.full(self.shape[1], global_sum, dtype=vector_array.dtype)
        if self.missing_variant_sample_indices.size > 0:
            missing_weights = vector_array[self.missing_variant_sample_indices]
            nonempty_missing = self.missing_variant_ptr[1:] > self.missing_variant_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.missing_variant_ptr[:-1][nonempty_missing]
                observed_sums[nonempty_missing] -= np.add.reduceat(missing_weights, starts)
        output -= np.asarray(self.means, dtype=vector_array.dtype) * observed_sums
        output /= np.asarray(self.scales, dtype=vector_array.dtype)
        return output

    def transpose_matmat(self, matrix: NDArray) -> NDArray:
        matrix_array = np.asarray(matrix, dtype=gpu_compute_numpy_dtype())
        if matrix_array.ndim != 2 or matrix_array.shape[0] != self.sample_count:
            raise ValueError("sample matrix must match sparse sample count.")
        if not np.any(matrix_array):
            return np.zeros((self.shape[1], matrix_array.shape[1]), dtype=matrix_array.dtype)
        output = np.zeros((self.shape[1], matrix_array.shape[1]), dtype=matrix_array.dtype)
        global_sum = np.sum(matrix_array, axis=0, dtype=matrix_array.dtype)
        if self.variant_sample_indices.size > 0:
            carrier_weights = self.variant_dosages.astype(matrix_array.dtype, copy=False)[:, None] * matrix_array[self.variant_sample_indices, :]
            nonempty_variants = self.variant_ptr[1:] > self.variant_ptr[:-1]
            if np.any(nonempty_variants):
                starts = self.variant_ptr[:-1][nonempty_variants]
                output[nonempty_variants, :] += np.add.reduceat(carrier_weights, starts, axis=0)
        observed_sums = np.broadcast_to(global_sum, output.shape).copy()
        if self.missing_variant_sample_indices.size > 0:
            missing_weights = matrix_array[self.missing_variant_sample_indices, :]
            nonempty_missing = self.missing_variant_ptr[1:] > self.missing_variant_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.missing_variant_ptr[:-1][nonempty_missing]
                observed_sums[nonempty_missing, :] -= np.add.reduceat(missing_weights, starts, axis=0)
        output -= np.asarray(self.means, dtype=matrix_array.dtype)[:, None] * observed_sums
        output /= np.asarray(self.scales, dtype=matrix_array.dtype)[:, None]
        return output

    def subset(self, local_variant_indices: NDArray) -> _SparseCarrierBackend:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved_local_indices.size == 0:
            empty_int = np.zeros(0, dtype=np.int32)
            empty_ptr = np.zeros(1, dtype=np.int64)
            return _SparseCarrierBackend(
                sample_count=self.sample_count,
                means=np.zeros(0, dtype=np.float32),
                scales=np.zeros(0, dtype=np.float32),
                variant_ptr=empty_ptr,
                variant_sample_indices=empty_int,
                variant_dosages=np.zeros(0, dtype=np.float32),
                missing_variant_ptr=empty_ptr.copy(),
                missing_variant_sample_indices=empty_int.copy(),
                sample_ptr=np.zeros(self.sample_count + 1, dtype=np.int64),
                sample_variant_indices=empty_int.copy(),
                sample_dosages=np.zeros(0, dtype=np.float32),
                sample_missing_ptr=np.zeros(self.sample_count + 1, dtype=np.int64),
                sample_missing_variant_indices=empty_int.copy(),
            )
        old_to_new = np.full(self.shape[1], -1, dtype=np.int32)
        old_to_new[resolved_local_indices] = np.arange(resolved_local_indices.shape[0], dtype=np.int32)
        selected_variant_sample_indices: list[NDArray] = []
        selected_variant_dosages: list[NDArray] = []
        variant_counts = np.zeros(resolved_local_indices.shape[0], dtype=np.int64)
        selected_missing_sample_indices: list[NDArray] = []
        missing_counts = np.zeros(resolved_local_indices.shape[0], dtype=np.int64)
        for new_variant_index, old_variant_index in enumerate(resolved_local_indices.tolist()):
            carrier_start = int(self.variant_ptr[old_variant_index])
            carrier_stop = int(self.variant_ptr[old_variant_index + 1])
            selected_variant_sample_indices.append(self.variant_sample_indices[carrier_start:carrier_stop].astype(np.int32, copy=False))
            selected_variant_dosages.append(self.variant_dosages[carrier_start:carrier_stop].astype(np.float32, copy=False))
            variant_counts[new_variant_index] = carrier_stop - carrier_start
            missing_start = int(self.missing_variant_ptr[old_variant_index])
            missing_stop = int(self.missing_variant_ptr[old_variant_index + 1])
            selected_missing_sample_indices.append(self.missing_variant_sample_indices[missing_start:missing_stop].astype(np.int32, copy=False))
            missing_counts[new_variant_index] = missing_stop - missing_start
        variant_sample_indices = (
            np.concatenate(selected_variant_sample_indices).astype(np.int32, copy=False)
            if selected_variant_sample_indices else np.zeros(0, dtype=np.int32)
        )
        variant_dosages = (
            np.concatenate(selected_variant_dosages).astype(np.float32, copy=False)
            if selected_variant_dosages else np.zeros(0, dtype=np.float32)
        )
        missing_variant_sample_indices = (
            np.concatenate(selected_missing_sample_indices).astype(np.int32, copy=False)
            if selected_missing_sample_indices else np.zeros(0, dtype=np.int32)
        )
        sample_mask = old_to_new[self.sample_variant_indices] >= 0
        sample_variant_indices = old_to_new[self.sample_variant_indices[sample_mask]].astype(np.int32, copy=False)
        sample_dosages = self.sample_dosages[sample_mask].astype(np.float32, copy=False)
        if sample_mask.any():
            kept_sample_ids = np.repeat(
                np.arange(self.sample_count, dtype=np.int32),
                np.diff(self.sample_ptr).astype(np.int32, copy=False),
            )[sample_mask]
            sample_counts = np.bincount(kept_sample_ids, minlength=self.sample_count).astype(np.int64, copy=False)
        else:
            sample_counts = np.zeros(self.sample_count, dtype=np.int64)
        sample_missing_mask = old_to_new[self.sample_missing_variant_indices] >= 0
        sample_missing_variant_indices = old_to_new[self.sample_missing_variant_indices[sample_missing_mask]].astype(np.int32, copy=False)
        if sample_missing_mask.any():
            kept_missing_sample_ids = np.repeat(
                np.arange(self.sample_count, dtype=np.int32),
                np.diff(self.sample_missing_ptr).astype(np.int32, copy=False),
            )[sample_missing_mask]
            sample_missing_counts = np.bincount(kept_missing_sample_ids, minlength=self.sample_count).astype(np.int64, copy=False)
        else:
            sample_missing_counts = np.zeros(self.sample_count, dtype=np.int64)
        variant_ptr = np.zeros(resolved_local_indices.shape[0] + 1, dtype=np.int64)
        variant_ptr[1:] = np.cumsum(variant_counts, dtype=np.int64)
        missing_variant_ptr = np.zeros(resolved_local_indices.shape[0] + 1, dtype=np.int64)
        missing_variant_ptr[1:] = np.cumsum(missing_counts, dtype=np.int64)
        sample_ptr = np.zeros(self.sample_count + 1, dtype=np.int64)
        sample_ptr[1:] = np.cumsum(sample_counts, dtype=np.int64)
        sample_missing_ptr = np.zeros(self.sample_count + 1, dtype=np.int64)
        sample_missing_ptr[1:] = np.cumsum(sample_missing_counts, dtype=np.int64)
        return _SparseCarrierBackend(
            sample_count=self.sample_count,
            means=np.asarray(self.means[resolved_local_indices], dtype=np.float32),
            scales=np.asarray(self.scales[resolved_local_indices], dtype=np.float32),
            variant_ptr=variant_ptr,
            variant_sample_indices=variant_sample_indices,
            variant_dosages=variant_dosages,
            missing_variant_ptr=missing_variant_ptr,
            missing_variant_sample_indices=missing_variant_sample_indices,
            sample_ptr=sample_ptr,
            sample_variant_indices=sample_variant_indices,
            sample_dosages=sample_dosages,
            sample_missing_ptr=sample_missing_ptr,
            sample_missing_variant_indices=sample_missing_variant_indices,
        )


def _build_sparse_backend(
    raw: Int8BatchCapable,
    raw_variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    sample_count: int,
) -> _SparseCarrierBackend:
    resolved_variant_indices = np.asarray(raw_variant_indices, dtype=np.int32)
    carrier_sample_chunks: list[NDArray] = []
    carrier_dosage_chunks: list[NDArray] = []
    carrier_counts = np.zeros(resolved_variant_indices.shape[0], dtype=np.int64)
    missing_sample_chunks: list[NDArray] = []
    missing_counts = np.zeros(resolved_variant_indices.shape[0], dtype=np.int64)
    local_start = 0
    batch_size = max(auto_batch_size_i8(sample_count), 1)
    for raw_batch in raw.iter_column_batches_i8(resolved_variant_indices, batch_size=batch_size):
        batch_values = np.asarray(raw_batch.values, dtype=np.int8)
        for local_batch_index in range(batch_values.shape[1]):
            local_variant_index = local_start + local_batch_index
            column = batch_values[:, local_batch_index]
            carrier_sample_indices = np.flatnonzero((column != PLINK_MISSING_INT8) & (column > 0)).astype(np.int32)
            missing_sample_indices = np.flatnonzero(column == PLINK_MISSING_INT8).astype(np.int32)
            carrier_sample_chunks.append(carrier_sample_indices)
            carrier_dosage_chunks.append(column[carrier_sample_indices].astype(np.float32, copy=False))
            missing_sample_chunks.append(missing_sample_indices)
            carrier_counts[local_variant_index] = carrier_sample_indices.shape[0]
            missing_counts[local_variant_index] = missing_sample_indices.shape[0]
        local_start += batch_values.shape[1]
    variant_sample_indices = (
        np.concatenate(carrier_sample_chunks).astype(np.int32, copy=False)
        if carrier_sample_chunks else np.zeros(0, dtype=np.int32)
    )
    variant_dosages = (
        np.concatenate(carrier_dosage_chunks).astype(np.float32, copy=False)
        if carrier_dosage_chunks else np.zeros(0, dtype=np.float32)
    )
    missing_variant_sample_indices = (
        np.concatenate(missing_sample_chunks).astype(np.int32, copy=False)
        if missing_sample_chunks else np.zeros(0, dtype=np.int32)
    )
    variant_ptr = np.zeros(resolved_variant_indices.shape[0] + 1, dtype=np.int64)
    variant_ptr[1:] = np.cumsum(carrier_counts, dtype=np.int64)
    missing_variant_ptr = np.zeros(resolved_variant_indices.shape[0] + 1, dtype=np.int64)
    missing_variant_ptr[1:] = np.cumsum(missing_counts, dtype=np.int64)
    carrier_variant_indices = np.repeat(
        np.arange(resolved_variant_indices.shape[0], dtype=np.int32),
        carrier_counts.astype(np.int32, copy=False),
    )
    if variant_sample_indices.size > 0:
        sample_order = np.argsort(variant_sample_indices, kind="stable")
        sample_variant_indices = carrier_variant_indices[sample_order].astype(np.int32, copy=False)
        sample_dosages = variant_dosages[sample_order].astype(np.float32, copy=False)
        sorted_sample_indices = variant_sample_indices[sample_order]
        sample_counts = np.bincount(sorted_sample_indices, minlength=sample_count).astype(np.int64, copy=False)
    else:
        sample_variant_indices = np.zeros(0, dtype=np.int32)
        sample_dosages = np.zeros(0, dtype=np.float32)
        sample_counts = np.zeros(sample_count, dtype=np.int64)
    sample_ptr = np.zeros(sample_count + 1, dtype=np.int64)
    sample_ptr[1:] = np.cumsum(sample_counts, dtype=np.int64)
    missing_variant_indices = np.repeat(
        np.arange(resolved_variant_indices.shape[0], dtype=np.int32),
        missing_counts.astype(np.int32, copy=False),
    )
    if missing_variant_sample_indices.size > 0:
        missing_sample_order = np.argsort(missing_variant_sample_indices, kind="stable")
        sample_missing_variant_indices = missing_variant_indices[missing_sample_order].astype(np.int32, copy=False)
        sorted_missing_sample_indices = missing_variant_sample_indices[missing_sample_order]
        sample_missing_counts = np.bincount(sorted_missing_sample_indices, minlength=sample_count).astype(np.int64, copy=False)
    else:
        sample_missing_variant_indices = np.zeros(0, dtype=np.int32)
        sample_missing_counts = np.zeros(sample_count, dtype=np.int64)
    sample_missing_ptr = np.zeros(sample_count + 1, dtype=np.int64)
    sample_missing_ptr[1:] = np.cumsum(sample_missing_counts, dtype=np.int64)
    return _SparseCarrierBackend(
        sample_count=sample_count,
        means=np.asarray(means[resolved_variant_indices], dtype=np.float32),
        scales=np.asarray(scales[resolved_variant_indices], dtype=np.float32),
        variant_ptr=variant_ptr,
        variant_sample_indices=variant_sample_indices,
        variant_dosages=variant_dosages,
        missing_variant_ptr=missing_variant_ptr,
        missing_variant_sample_indices=missing_variant_sample_indices,
        sample_ptr=sample_ptr,
        sample_variant_indices=sample_variant_indices,
        sample_dosages=sample_dosages,
        sample_missing_ptr=sample_missing_ptr,
        sample_missing_variant_indices=sample_missing_variant_indices,
    )


# Module-level set of (label, exception-type-name) tuples that have already
# triggered a bitpacked-dispatch fallback warning. Used to log once per unique
# failure mode rather than on every HVP in the TR-Newton CG inner loop.
_BITPACKED_DISPATCH_FALLBACK_WARNED: set[tuple[str, str]] = set()


def _warn_bitpacked_dispatch_fallback(label: str, exc: BaseException) -> None:
    """Emit a one-shot warning when the bitpacked SGM fast path falls back to streaming.

    Without this guard, ``except Exception: pass`` in the bitpacked dispatch
    can silently degrade the whole fit from sub-second HVPs to streaming
    (~9 hours) with zero visibility. We log once per (label, exception-type)
    pair so the user sees the failure but the inner loop stays quiet.
    """
    exc_type_name = type(exc).__name__
    key = (label, exc_type_name)
    if key in _BITPACKED_DISPATCH_FALLBACK_WARNED:
        return
    _BITPACKED_DISPATCH_FALLBACK_WARNED.add(key)
    try:
        log(
            f"bitpacked SGM dispatch fell back to streaming ({label}, "
            f"exc {exc_type_name}: {exc}). Subsequent identical failures will be silent."
        )
    except Exception:  # noqa: BLE001 - logging must never break the fast-path fallback
        pass


def _sgm_variant_indices_is_identity(sgm: StandardizedGenotypeMatrix) -> bool:
    """Return whether ``sgm.variant_indices`` equals ``np.arange(sgm.shape[1])``.

    Caches the result on the SGM instance keyed by ``id(variant_indices)`` so
    a reused SGM whose ``variant_indices`` is reassigned (different array
    object) transparently recomputes. The check is otherwise O(p) per call,
    which is wasted work inside TR-Newton CG hot loops where the SGM and its
    indices are immutable for thousands of HVPs.
    """
    variant_idx_arr = np.asarray(sgm.variant_indices)
    key = id(sgm.variant_indices)
    cached = getattr(sgm, "_variant_indices_is_identity_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    is_identity = bool(
        variant_idx_arr.shape[0] == sgm.shape[1]
        and np.array_equal(variant_idx_arr, np.arange(sgm.shape[1]))
    )
    try:
        sgm._variant_indices_is_identity_cache = (key, is_identity)
    except AttributeError:  # slots SGM that hasn't declared the cache slot — be safe.
        pass
    return is_identity


@dataclass(slots=True)
class StandardizedGenotypeMatrix:
    """A genotype matrix that applies z-score standardization on the fly.

    For each variant j: standardized_value = (raw_dosage - mean_j) / scale_j
    Missing values (NaN) are imputed to the mean (producing 0 after centering).

    GPU acceleration uses CuPy (cuBLAS) for matmul, bypassing JAX/XLA which
    has known compilation bugs on some GPU architectures.  Falls back to numpy BLAS on CPU.
    """
    raw: RawGenotypeMatrix | None
    means: NDArray       # per-variant mean from training data
    scales: NDArray      # per-variant std dev from training data
    variant_indices: NDArray  # which columns of raw to use (for subsetting)
    support_counts: NDArray | None = None  # non-zero dosage count per source variant
    sample_count: int | None = field(default=None, repr=False)
    _enable_hybrid_backend: bool = field(default=True, repr=False)
    _dense_cache: NDArray | None = field(init=False, default=None, repr=False)
    _cupy_cache: Any | None = field(init=False, default=None, repr=False)  # cupy.ndarray
    _jax_cache: jax.Array | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache: Any | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache_local_indices: NDArray | None = field(init=False, default=None, repr=False)
    _local_cache_directory: tempfile.TemporaryDirectory[str] | None = field(init=False, default=None, repr=False)
    _dense_backend: StandardizedGenotypeMatrix | None = field(init=False, default=None, repr=False)
    _sparse_backend: _SparseCarrierBackend | None = field(init=False, default=None, repr=False)
    _dense_local_lookup: NDArray | None = field(init=False, default=None, repr=False)
    _sparse_local_lookup: NDArray | None = field(init=False, default=None, repr=False)
    _sample_space_nystrom_basis_cpu_cache: dict[tuple[int, int], NDArray] = field(init=False, default_factory=dict, repr=False)
    _sample_space_nystrom_basis_gpu_cache: dict[tuple[int, int], Any] = field(init=False, default_factory=dict, repr=False)
    _sample_space_probe_projection_cache: dict[tuple[int, int], NDArray] = field(init=False, default_factory=dict, repr=False)
    _sample_space_probe_projection_gpu_cache: dict[tuple[int, int], Any] = field(init=False, default_factory=dict, repr=False)
    _sample_space_cpu_preconditioner_cache: Any | None = field(init=False, default=None, repr=False)
    _sample_space_gpu_preconditioner_cache: Any | None = field(init=False, default=None, repr=False)
    # Parent-lifetime invariant: a GPU subset cache produced by ``subset()`` is a
    # view/slice into the parent's underlying CuPy buffer (see
    # ``_cupy_cache_subset_columns``). If the parent ``StandardizedGenotypeMatrix``
    # is garbage-collected before the subset is consumed, that buffer is freed
    # and the subset becomes a dangling GPU pointer. Holding a strong reference
    # to the parent here keeps it alive for the subset's full lifetime.
    _parent_genotype_matrix: StandardizedGenotypeMatrix | None = field(init=False, default=None, repr=False)
    _n_samples: int = field(init=False, default=0, repr=False)
    # Phase 4 LD-block / N-GPU wiring: opt-in partition + GPU scheduler.
    # Both default to None so legacy entry points (no use_ld_blocks flag) see
    # exactly the prior behavior. They are populated by ``BayesianPGS.fit``
    # (and mirrored onto the validation matrix view) when
    # ``ModelConfig.use_ld_blocks=True``.
    _ld_block_partition: Any | None = field(init=False, default=None, repr=False)
    _ld_block_scheduler: Any | None = field(init=False, default=None, repr=False)
    # Cache for variant_indices identity check used by the bitpacked fast path
    # in matvec_numpy / transpose_matvec_numpy. Tuple of (id(variant_indices),
    # is_identity_bool) — recomputed lazily if variant_indices is reassigned.
    _variant_indices_is_identity_cache: tuple[int, bool] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        if self.means.ndim != 1 or self.scales.ndim != 1 or self.means.shape != self.scales.shape:
            raise ValueError("means and scales must be matching 1D arrays.")
        if self.support_counts is not None:
            self.support_counts = np.asarray(self.support_counts, dtype=np.int32)
            if self.support_counts.ndim != 1 or self.support_counts.shape != self.means.shape:
                raise ValueError("support_counts must match means/scales shape.")
        self.variant_indices = np.asarray(self.variant_indices, dtype=np.int32)
        source_variant_count = int(self.means.shape[0])
        if self.raw is not None:
            self._n_samples = int(self.raw.shape[0])
            if int(self.raw.shape[1]) != source_variant_count:
                raise ValueError("raw genotype matrix width must match means/scales.")
        elif self.sample_count is not None:
            self._n_samples = int(self.sample_count)
        else:
            self._n_samples = 0
        if self.variant_indices.ndim != 1:
            raise ValueError("variant_indices must be 1D.")
        if np.any(self.variant_indices < 0) or np.any(self.variant_indices >= source_variant_count):
            raise ValueError("variant_indices out of bounds.")
        self._configure_operator_backend()

    def _configure_operator_backend(self) -> None:
        self._dense_backend = None
        self._sparse_backend = None
        self._dense_local_lookup = None
        self._sparse_local_lookup = None
        if (
            not self._enable_hybrid_backend
            or self.raw is None
            or self.support_counts is None
            or not _supports_int8_batches(self.raw)
            or self.shape[1] == 0
        ):
            return
        selected_support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
        sparse_local_indices = np.flatnonzero(selected_support_counts <= HYBRID_SPARSE_SUPPORT_THRESHOLD).astype(np.int32)
        if sparse_local_indices.shape[0] < HYBRID_SPARSE_MIN_VARIANT_COUNT:
            return
        dense_mask = np.ones(self.shape[1], dtype=bool)
        dense_mask[sparse_local_indices] = False
        dense_local_indices = np.flatnonzero(dense_mask).astype(np.int32)
        self._sparse_backend = _build_sparse_backend(
            raw=self.raw,
            raw_variant_indices=self.variant_indices[sparse_local_indices],
            means=self.means,
            scales=self.scales,
            sample_count=self.shape[0],
        )
        sparse_lookup = np.full(self.shape[1], -1, dtype=np.int32)
        sparse_lookup[sparse_local_indices] = np.arange(sparse_local_indices.shape[0], dtype=np.int32)
        self._sparse_local_lookup = sparse_lookup
        if dense_local_indices.shape[0] > 0:
            dense_raw = cast(RawGenotypeMatrix, self.raw)
            self._dense_backend = StandardizedGenotypeMatrix(
                raw=dense_raw,
                means=self.means,
                scales=self.scales,
                variant_indices=self.variant_indices[dense_local_indices],
                support_counts=self.support_counts,
                sample_count=self.shape[0],
                _enable_hybrid_backend=False,
            )
            dense_lookup = np.full(self.shape[1], -1, dtype=np.int32)
            dense_lookup[dense_local_indices] = np.arange(dense_local_indices.shape[0], dtype=np.int32)
            self._dense_local_lookup = dense_lookup
        log(
            "    hybrid standardized operator: "
            + f"{sparse_local_indices.shape[0]} sparse variants + {dense_local_indices.shape[0]} dense variants  mem={mem()}"
        )

    def _uses_hybrid_backend(self) -> bool:
        return self._sparse_backend is not None

    def _hybrid_local_components(
        self,
        local_variant_indices: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if not self._uses_hybrid_backend():
            empty = np.zeros(0, dtype=np.int32)
            empty_mask = np.zeros(resolved_local_indices.shape[0], dtype=bool)
            return empty_mask, empty, empty_mask.copy(), empty
        sparse_lookup = (
            np.full(self.shape[1], -1, dtype=np.int32)
            if self._sparse_local_lookup is None
            else self._sparse_local_lookup
        )
        dense_lookup = (
            np.full(self.shape[1], -1, dtype=np.int32)
            if self._dense_local_lookup is None
            else self._dense_local_lookup
        )
        sparse_child_local_indices = sparse_lookup[resolved_local_indices]
        dense_child_local_indices = dense_lookup[resolved_local_indices]
        sparse_mask = sparse_child_local_indices >= 0
        dense_mask = dense_child_local_indices >= 0
        if np.any(~(sparse_mask | dense_mask)):
            raise RuntimeError("hybrid standardized operator lost local column mapping.")
        return (
            sparse_mask,
            sparse_child_local_indices[sparse_mask].astype(np.int32, copy=False),
            dense_mask,
            dense_child_local_indices[dense_mask].astype(np.int32, copy=False),
        )

    def _hybrid_parent_local_indices(self) -> tuple[NDArray, NDArray]:
        if not self._uses_hybrid_backend():
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        sparse_parent_local_indices = (
            np.flatnonzero(self._sparse_local_lookup >= 0).astype(np.int32)
            if self._sparse_local_lookup is not None
            else np.zeros(0, dtype=np.int32)
        )
        dense_parent_local_indices = (
            np.flatnonzero(self._dense_local_lookup >= 0).astype(np.int32)
            if self._dense_local_lookup is not None
            else np.zeros(0, dtype=np.int32)
        )
        return sparse_parent_local_indices, dense_parent_local_indices

    def _materialize_hybrid_columns(
        self,
        local_variant_indices: NDArray,
        *,
        batch_size: int,
    ) -> NDArray:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        output = np.empty((self.shape[0], resolved_local_indices.shape[0]), dtype=np.float32)
        if resolved_local_indices.size == 0:
            return output
        sparse_mask, sparse_child_local_indices, dense_mask, dense_child_local_indices = self._hybrid_local_components(
            resolved_local_indices
        )
        if np.any(sparse_mask):
            if self._sparse_backend is None:
                raise RuntimeError("hybrid sparse backend is not configured.")
            output[:, sparse_mask] = self._sparse_backend.materialize_columns(sparse_child_local_indices)
        if np.any(dense_mask):
            if self._dense_backend is None:
                raise RuntimeError("hybrid dense backend is not configured.")
            output[:, dense_mask] = self._dense_backend.subset(dense_child_local_indices).materialize(batch_size=batch_size)
        return output

    @property
    def _gpu_cache(self) -> Any | None:
        return self._cupy_cache

    @_gpu_cache.setter
    def _gpu_cache(self, value: Any | None) -> None:
        self._cupy_cache = value

    @property
    def shape(self) -> tuple[int, int]:
        return self._n_samples, int(self.variant_indices.shape[0])

    def dense_bytes(self) -> int:
        """Estimated bytes if materialized as float32."""
        return int(self.shape[0]) * int(self.shape[1]) * 4

    def release_raw_storage(self) -> None:
        """Drop the backing raw matrix once a dense or GPU cache exists."""
        if self._dense_cache is None and self._cupy_cache is None:
            raise RuntimeError("cannot release raw storage before materializing genotype data.")
        self.raw = None
        self._local_cache_directory = None
        self._cupy_subset_cache = None
        self._cupy_subset_cache_local_indices = None
        if self._dense_backend is not None:
            self._dense_backend.raw = None
            self._dense_backend._local_cache_directory = None

    def clear_sample_space_nystrom_cache(self) -> None:
        self._sample_space_nystrom_basis_cpu_cache.clear()
        self._sample_space_nystrom_basis_gpu_cache.clear()
        self._sample_space_probe_projection_cache.clear()
        self._sample_space_probe_projection_gpu_cache.clear()
        self._sample_space_cpu_preconditioner_cache = None
        self._sample_space_gpu_preconditioner_cache = None
        self._cupy_subset_cache = None
        self._cupy_subset_cache_local_indices = None

    def _sharded_gpu_plans(
        self,
        *,
        cupy: Any,
        splits: Sequence[tuple[int, int, int]],
        dtype: Any,
        backend: str,
        metadata_bytes_per_column: int = 0,
    ) -> tuple[_GpuMaterializationMemoryPlan, ...] | None:
        plans: list[_GpuMaterializationMemoryPlan] = []
        for device_id, start, stop in splits:
            width = int(stop) - int(start)
            with _cupy_device_context(cupy, device_id):
                plan = _estimate_gpu_materialization_memory_plan(
                    n_rows=self.shape[0],
                    n_cols=width,
                    dtype=dtype,
                    backend=backend,
                    cupy=cupy,
                    metadata_bytes=width * int(metadata_bytes_per_column),
                )
            _log_gpu_materialization_memory_plan(plan)
            if not plan.fits:
                _warn_gpu_materialization_unavailable(
                    f"multi-GPU shard {device_id} {backend} cache exceeds budget",
                    plan,
                )
                return None
            plans.append(plan)
        return tuple(plans)

    def _try_materialize_gpu_sharded(
        self,
        *,
        cupy: Any,
        use_int8_gpu_cache: bool,
    ) -> bool:
        device_ids = _cupy_device_ids(cupy)
        if len(device_ids) < 2 or self.shape[1] < 2:
            return False
        splits = _split_contiguous_columns(self.shape[1], device_ids)
        if len(splits) < 2:
            return False

        backend: str | None = None
        dtype: Any | None = None
        plans: tuple[_GpuMaterializationMemoryPlan, ...] | None = None
        metadata_bytes_per_column = 0
        if self._dense_cache is not None:
            backend = "fp32-resident"
            dtype = np.float32
            plans = self._sharded_gpu_plans(cupy=cupy, splits=splits, dtype=dtype, backend=backend)
        elif use_int8_gpu_cache:
            fp16_dtype = getattr(cupy, "float16", None)
            if (
                GPU_FP16_RESIDENT_CACHE_ENABLED
                and fp16_dtype is not None
                and _cupy_compute_dtype(cupy) == cupy.float32
            ):
                plans = self._sharded_gpu_plans(
                    cupy=cupy,
                    splits=splits,
                    dtype=fp16_dtype,
                    backend="fp16-resident",
                )
                if plans is not None:
                    backend = "fp16-resident"
                    dtype = fp16_dtype
            if plans is None:
                metadata_bytes_per_column = np.dtype(np.float32).itemsize * 2
                plans = self._sharded_gpu_plans(
                    cupy=cupy,
                    splits=splits,
                    dtype=np.int8,
                    backend="int8-resident",
                    metadata_bytes_per_column=metadata_bytes_per_column,
                )
                if plans is not None:
                    backend = "int8-resident"
                    dtype = np.int8
        else:
            backend = "fp32-resident"
            dtype = np.float32
            plans = self._sharded_gpu_plans(cupy=cupy, splits=splits, dtype=dtype, backend=backend)

        if plans is None or backend is None or dtype is None:
            return False

        log(
            "    uploading standardized genotypes to multi-GPU sharded cache "
            + f"({len(splits)} GPUs, backend={backend})  mem={mem()}"
        )
        shards: list[_CupyDeviceCacheShard] = []
        raw_matrix = self.raw
        dense_ref = self._dense_cache
        try:
            if backend == "fp32-resident" and dense_ref is not None:
                for device_id, start, stop in splits:
                    with _cupy_device_context(cupy, device_id):
                        shard_cache = cupy.asarray(dense_ref[:, start:stop], dtype=cupy.float32)
                        _cupy_device_synchronize(cupy, device_id)
                    shards.append(_CupyDeviceCacheShard(device_id=device_id, column_start=start, cache=shard_cache))
                self._dense_cache = None
            elif backend == "fp16-resident":
                if raw_matrix is None:
                    raise RuntimeError("fp16 multi-GPU cache requires raw backing storage.")
                raw_int8 = cast(Int8BatchCapable, raw_matrix)
                for device_id, start, stop in splits:
                    width = int(stop) - int(start)
                    with _cupy_device_context(cupy, device_id):
                        shard_cache = cupy.empty((self.shape[0], width), dtype=cupy.float16, order="F")
                        _upload_standardized_int8_tiles_overlapped(
                            cupy=cupy,
                            raw_int8=raw_int8,
                            variant_indices=self.variant_indices[start:stop],
                            means=self.means,
                            scales=self.scales,
                            gpu_destination=shard_cache,
                            sample_count=int(self.shape[0]),
                            upload_batch_size=auto_batch_size_i8(self.shape[0]),
                            standardized_dtype=cupy.float16,
                        )
                        _cupy_device_synchronize(cupy, device_id)
                    shards.append(_CupyDeviceCacheShard(device_id=device_id, column_start=start, cache=shard_cache))
            elif backend == "int8-resident":
                if raw_matrix is None:
                    raise RuntimeError("int8 multi-GPU cache requires raw backing storage.")
                raw_int8 = cast(Int8BatchCapable, raw_matrix)
                gpu_int8_dtype = cupy.int8 if hasattr(cupy, "int8") else np.int8
                if isinstance(raw_matrix, Int8RawGenotypeMatrix):
                    _madvise_willneed_array(raw_matrix.matrix)
                for (device_id, start, stop), plan in zip(splits, plans, strict=True):
                    width = int(stop) - int(start)
                    shard_variant_indices = self.variant_indices[start:stop]
                    with _cupy_device_context(cupy, device_id):
                        full_int8_block = None
                        int8_total_bytes = int(self.shape[0]) * width
                        if int8_total_bytes <= int(plan.budget_bytes * INT8_ONE_SHOT_GPU_BUDGET_FRACTION):
                            full_int8_block = _read_int8_columns_one_shot(raw_matrix, shard_variant_indices)
                        if full_int8_block is not None:
                            shard_raw = cupy.asarray(full_int8_block, dtype=gpu_int8_dtype)
                            if not shard_raw.flags.f_contiguous:
                                if hasattr(cupy, "asfortranarray"):
                                    shard_raw = cupy.asfortranarray(shard_raw)
                                else:
                                    shard_raw = np.asfortranarray(np.asarray(shard_raw, dtype=gpu_int8_dtype))
                        else:
                            shard_raw = cupy.empty((self.shape[0], width), dtype=gpu_int8_dtype, order="F")
                            _upload_int8_tiles_overlapped(
                                cupy=cupy,
                                raw_int8=raw_int8,
                                variant_indices=shard_variant_indices,
                                gpu_destination=shard_raw,
                                sample_count=int(self.shape[0]),
                                upload_batch_size=auto_batch_size_i8(self.shape[0]),
                                gpu_int8_dtype=gpu_int8_dtype,
                            )
                        shard_cache = _CupyInt8StandardizedCache(
                            raw_values=shard_raw,
                            means=cupy.asarray(self.means[shard_variant_indices], dtype=cupy.float32),
                            scales=cupy.asarray(self.scales[shard_variant_indices], dtype=cupy.float32),
                            cupy=cupy,
                        )
                        _cupy_device_synchronize(cupy, device_id)
                    shards.append(_CupyDeviceCacheShard(device_id=device_id, column_start=start, cache=shard_cache))
            else:
                if raw_matrix is None:
                    raise RuntimeError("multi-GPU materialization requires raw backing storage or a dense cache.")
                for device_id, start, stop in splits:
                    width = int(stop) - int(start)
                    with _cupy_device_context(cupy, device_id):
                        shard_cache = cupy.empty((self.shape[0], width), dtype=cupy.float32, order="F")
                        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                            raw_matrix,
                            self.variant_indices[start:stop],
                            self.means,
                            self.scales,
                            batch_size=auto_batch_size(self.shape[0]),
                            cupy=cupy,
                        ):
                            shard_cache[:, batch_slice] = standardized_batch
                        _cupy_device_synchronize(cupy, device_id)
                    shards.append(_CupyDeviceCacheShard(device_id=device_id, column_start=start, cache=shard_cache))
            self._cupy_cache = _CupyShardedStandardizedCache(tuple(shards), cupy=cupy)
            log(
                "    CuPy multi-GPU matrix ready "
                + f"({len(shards)} shards, {_cupy_cache_nbytes(self._cupy_cache) / 1e9:.1f} GB)  mem={mem()}"
            )
            return True
        except BaseException:
            self._cupy_cache = None
            if dense_ref is not None:
                self._dense_cache = dense_ref
            for device_id in device_ids:
                with _cupy_device_context(cupy, device_id):
                    _release_cupy_cached_memory(cupy)
            raise

    def try_materialize_gpu(self) -> bool:
        """Materialize the standardized matrix onto GPU memory when possible."""
        if self._cupy_cache is not None:
            return True
        cupy = _try_import_cupy()
        if cupy is None:
            return False
        active_plan: _GpuMaterializationMemoryPlan | None = None
        try:
            use_int8_gpu_cache = self._dense_cache is None and self.raw is not None and _supports_int8_batches(self.raw)
            if len(_cupy_device_ids(cupy)) >= 2:
                return self._try_materialize_gpu_sharded(
                    cupy=cupy,
                    use_int8_gpu_cache=use_int8_gpu_cache,
                )
            metadata_bytes = int(self.shape[1]) * np.dtype(np.float32).itemsize * 2
            if self._dense_cache is not None:
                active_plan = _estimate_gpu_materialization_memory_plan(
                    n_rows=self.shape[0],
                    n_cols=self.shape[1],
                    dtype=np.float32,
                    backend="fp32-resident",
                    cupy=cupy,
                )
                _log_gpu_materialization_memory_plan(active_plan)
                if not active_plan.fits:
                    _warn_gpu_materialization_unavailable("fp32 cache exceeds budget", active_plan)
                    return False
                nbytes = active_plan.required_bytes
            elif use_int8_gpu_cache:
                fp16_dtype = getattr(cupy, "float16", None)
                if (
                    GPU_FP16_RESIDENT_CACHE_ENABLED
                    and fp16_dtype is not None
                    and _cupy_compute_dtype(cupy) == cupy.float32
                ):
                    fp16_plan = _estimate_gpu_materialization_memory_plan(
                        n_rows=self.shape[0],
                        n_cols=self.shape[1],
                        dtype=fp16_dtype,
                        backend="fp16-resident",
                        cupy=cupy,
                    )
                    _log_gpu_materialization_memory_plan(fp16_plan)
                    if fp16_plan.fits:
                        active_plan = fp16_plan
                        nbytes = active_plan.required_bytes
                    else:
                        log("    fp16 GPU materialization plan does not fit; evaluating int8-resident cache")
                        active_plan = None
                if active_plan is None:
                    active_plan = _estimate_gpu_materialization_memory_plan(
                        n_rows=self.shape[0],
                        n_cols=self.shape[1],
                        dtype=np.int8,
                        backend="int8-resident",
                        cupy=cupy,
                        metadata_bytes=metadata_bytes,
                    )
                    _log_gpu_materialization_memory_plan(active_plan)
                    if not active_plan.fits:
                        _warn_gpu_materialization_unavailable("int8 cache exceeds budget", active_plan)
                        return False
                    nbytes = active_plan.required_bytes
            else:
                active_plan = _estimate_gpu_materialization_memory_plan(
                    n_rows=self.shape[0],
                    n_cols=self.shape[1],
                    dtype=np.float32,
                    backend="fp32-resident",
                    cupy=cupy,
                )
                _log_gpu_materialization_memory_plan(active_plan)
                if not active_plan.fits:
                    _warn_gpu_materialization_unavailable("fp32 cache exceeds budget", active_plan)
                    return False
                nbytes = active_plan.required_bytes
            budget_bytes = active_plan.budget_bytes
            if self._dense_cache is not None:
                log(f"    uploading RAM-resident matrix to GPU ({nbytes / 1e9:.1f} GB)  mem={mem()}")
                # Defer-and-restore on OOM: keep ``self._dense_cache`` populated
                # until *after* the async H2D copy has been synchronized. If the
                # device-side allocation or copy raises an out-of-memory error
                # (which can surface asynchronously at synchronize time), we
                # restore the host cache so callers can still fall back to
                # host-resident computation instead of being left with neither
                # a GPU nor a dense cache.
                dense_ref = self._dense_cache
                try:
                    self._cupy_cache = cupy.asarray(dense_ref)
                    cupy.cuda.Device().synchronize()
                except BaseException:
                    self._cupy_cache = None
                    self._dense_cache = dense_ref
                    raise
                self._dense_cache = None
                del dense_ref
            elif use_int8_gpu_cache and active_plan.backend == "fp16-resident":
                log(f"    uploading standardized genotypes to GPU fp16 cache ({nbytes / 1e9:.1f} GB)  mem={mem()}")
                raw_matrix = self.raw
                if raw_matrix is None:
                    raise RuntimeError("fp16 GPU cache requires raw backing storage.")
                gpu_matrix = cupy.empty(self.shape, dtype=cupy.float16, order="F")
                _upload_standardized_int8_tiles_overlapped(
                    cupy=cupy,
                    raw_int8=cast(Int8BatchCapable, raw_matrix),
                    variant_indices=self.variant_indices,
                    means=self.means,
                    scales=self.scales,
                    gpu_destination=gpu_matrix,
                    sample_count=int(self.shape[0]),
                    upload_batch_size=auto_batch_size_i8(self.shape[0]),
                    standardized_dtype=cupy.float16,
                )
                cupy.cuda.Device().synchronize()
                self._cupy_cache = gpu_matrix
            elif use_int8_gpu_cache:
                log(f"    uploading raw int8 genotypes to GPU ({nbytes / 1e9:.1f} GB incl. scales)  mem={mem()}")
                raw_matrix = self.raw
                if raw_matrix is None:
                    raise RuntimeError("raw int8 GPU cache requires raw backing storage.")
                raw_int8 = cast(Int8BatchCapable, raw_matrix)
                gpu_int8_dtype = cupy.int8 if hasattr(cupy, "int8") else np.int8
                if isinstance(raw_matrix, Int8RawGenotypeMatrix):
                    _madvise_willneed_array(raw_matrix.matrix)
                # Fastest path: if the leaf is a single F-order int8 numpy memmap
                # (the typical state after ``try_cache_persistently``), upload it
                # via N parallel pinned-buffer reads + N async H2D streams. This
                # bypasses the per-batch iterator and saturates disk queue depth
                # + PCIe simultaneously. Bit-identical to the tiled path.
                if isinstance(raw_matrix, Int8RawGenotypeMatrix) and isinstance(raw_matrix.matrix, np.memmap):
                    parallel_dst = cupy.empty(self.shape, dtype=gpu_int8_dtype, order="F")
                    if _try_upload_int8_parallel_memmap(
                        cupy=cupy,
                        raw=raw_matrix,
                        variant_indices=self.variant_indices,
                        gpu_destination=parallel_dst,
                        sample_count=int(self.shape[0]),
                    ):
                        gpu_matrix = parallel_dst
                        cupy.cuda.Device().synchronize()
                        self._cupy_cache = _CupyInt8StandardizedCache(
                            raw_values=gpu_matrix,
                            means=cupy.asarray(self.means[self.variant_indices], dtype=cupy.float32),
                            scales=cupy.asarray(self.scales[self.variant_indices], dtype=cupy.float32),
                            cupy=cupy,
                        )
                        cupy.cuda.Device().synchronize()
                        import gc
                        gc.collect()
                        log(f"    CuPy GPU matrix ready ({_cupy_cache_nbytes(self._cupy_cache) / 1e9:.1f} GB)  mem={mem()}")
                        return True
                    del parallel_dst
                # Fast path: if the entire int8 block fits inside the GPU cache
                # budget, prefer one contiguous raw read plus one H2D copy over
                # Python tile iteration. AoU fit-stage caches are Fortran-order
                # int8 mmaps, so this turns a minutes-long tiled upload into one
                # sequential mmap pass and one device allocation.
                int8_total_bytes = int(self.shape[0]) * int(self.shape[1])
                if int8_total_bytes <= int(budget_bytes * INT8_ONE_SHOT_GPU_BUDGET_FRACTION):
                    full_int8_block = _read_int8_columns_one_shot(raw_matrix, self.variant_indices)
                    if full_int8_block is not None:
                        log(
                            "    raw int8 block fits one-shot upload budget; "
                            + f"uploading {self.shape[1]} variants in one H2D copy  mem={mem()}"
                        )
                        gpu_matrix = cupy.asarray(full_int8_block, dtype=gpu_int8_dtype)
                        if not gpu_matrix.flags.f_contiguous:
                            if hasattr(cupy, "asfortranarray"):
                                gpu_matrix = cupy.asfortranarray(gpu_matrix)
                            else:
                                gpu_matrix = np.asfortranarray(np.asarray(gpu_matrix, dtype=gpu_int8_dtype))
                    else:
                        gpu_matrix = cupy.empty(self.shape, dtype=gpu_int8_dtype, order="F")
                        _upload_int8_tiles_overlapped(
                            cupy=cupy,
                            raw_int8=raw_int8,
                            variant_indices=self.variant_indices,
                            gpu_destination=gpu_matrix,
                            sample_count=int(self.shape[0]),
                            upload_batch_size=auto_batch_size_i8(self.shape[0]),
                            gpu_int8_dtype=gpu_int8_dtype,
                        )
                else:
                    gpu_matrix = cupy.empty(self.shape, dtype=gpu_int8_dtype, order="F")
                    _upload_int8_tiles_overlapped(
                        cupy=cupy,
                        raw_int8=raw_int8,
                        variant_indices=self.variant_indices,
                        gpu_destination=gpu_matrix,
                        sample_count=int(self.shape[0]),
                        upload_batch_size=auto_batch_size_i8(self.shape[0]),
                        gpu_int8_dtype=gpu_int8_dtype,
                    )
                # Synchronization invariant: callers may read ``self._cupy_cache``
                # immediately after ``try_materialize_gpu()`` returns, so make sure
                # every async H2D copy issued above has completed before we publish
                # the cache reference.
                cupy.cuda.Device().synchronize()
                self._cupy_cache = _CupyInt8StandardizedCache(
                    raw_values=gpu_matrix,
                    means=cupy.asarray(self.means[self.variant_indices], dtype=cupy.float32),
                    scales=cupy.asarray(self.scales[self.variant_indices], dtype=cupy.float32),
                    cupy=cupy,
                )
            else:
                log(f"    streaming {self.shape[1]} variants x {self.shape[0]} samples ({nbytes / 1e9:.1f} GB) directly to GPU  mem={mem()}")
                gpu_matrix = cupy.empty(self.shape, dtype=cupy.float32, order="F")
                if self.raw is None:
                    raise RuntimeError("GPU materialization requires raw backing storage or a dense cache.")
                # For GPU materialization, maximize batch size to minimize Python
                # overhead and kernel launch latency. The working memory per batch is
                # ~5 bytes/element (int8 + float32), so we can fit larger batches than
                # the CPU-oriented BED_READER_TARGET_BATCH_BYTES assumes.
                gpu_working_bytes = int(self.shape[0]) * int(self.shape[1]) * 5
                if gpu_working_bytes < _gpu_materialization_budget_bytes(cupy) * 0.5:
                    upload_batch_size = self.shape[1]  # entire block in one batch
                elif _supports_int8_batches(self.raw):
                    upload_batch_size = auto_batch_size_i8(self.shape[0])
                else:
                    upload_batch_size = auto_batch_size(self.shape[0])
                if _supports_int8_batches(self.raw):
                    _upload_standardized_int8_tiles_overlapped(
                        cupy=cupy,
                        raw_int8=self.raw,
                        variant_indices=self.variant_indices,
                        means=self.means,
                        scales=self.scales,
                        gpu_destination=gpu_matrix,
                        sample_count=int(self.shape[0]),
                        upload_batch_size=int(upload_batch_size),
                        standardized_dtype=cupy.float32,
                    )
                else:
                    raw_matrix = cast(RawGenotypeMatrix, self.raw)
                    for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                        raw_matrix,
                        self.variant_indices,
                        self.means,
                        self.scales,
                        batch_size=upload_batch_size,
                        cupy=cupy,
                    ):
                        gpu_matrix[:, batch_slice] = standardized_batch
                # Synchronization invariant: callers may read ``self._cupy_cache``
                # immediately after ``try_materialize_gpu()`` returns, so make sure
                # every async H2D copy issued above has completed before we publish
                # the cache reference.
                cupy.cuda.Device().synchronize()
                self._cupy_cache = gpu_matrix
            cupy.cuda.Device().synchronize()
            import gc
            gc.collect()
            log(f"    CuPy GPU matrix ready ({_cupy_cache_nbytes(self._cupy_cache) / 1e9:.1f} GB)  mem={mem()}")
            return True
        except (MemoryError, OSError, RuntimeError) as exc:
            self._cupy_cache = None
            if not _is_cupy_out_of_memory(exc):
                raise
            _release_cupy_cached_memory(cupy)
            log(f"    CuPy GPU upload failed ({exc})  mem={mem()}")
            if active_plan is not None:
                _warn_gpu_materialization_unavailable(f"allocation failed: {exc}", active_plan)
            return False

    def try_materialize_gpu_subset(
        self,
        local_variant_indices: Sequence[int] | NDArray,
    ) -> Any | None:
        """Materialize a selected standardized column subset onto GPU memory when possible."""
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved_local_indices.size == 0:
            return None
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                raise RuntimeError("CuPy cache requires CuPy runtime.")
            if _cupy_cache_is_int8_standardized(self._cupy_cache) or _cupy_cache_is_sharded(self._cupy_cache):
                return _cupy_cache_standardized_columns(
                    self._cupy_cache,
                    resolved_local_indices,
                    cupy=cupy,
                    dtype=cupy.float32,
                )
            return self._cupy_cache[:, resolved_local_indices]
        if (
            self._cupy_subset_cache is not None
            and self._cupy_subset_cache_local_indices is not None
            and np.array_equal(self._cupy_subset_cache_local_indices, resolved_local_indices)
        ):
            return self._cupy_subset_cache

        cupy = _try_import_cupy()
        if cupy is None:
            return None
        subset_plan = _estimate_gpu_materialization_memory_plan(
            n_rows=self.shape[0],
            n_cols=int(resolved_local_indices.shape[0]),
            dtype=np.float32,
            backend="fp32-subset",
            cupy=cupy,
        )
        _log_gpu_materialization_memory_plan(subset_plan)
        nbytes = subset_plan.required_bytes
        if not subset_plan.fits:
            _warn_gpu_materialization_unavailable("GPU subset exceeds budget", subset_plan)
            return None
        try:
            if self._dense_cache is not None:
                gpu_subset = cupy.asarray(self._dense_cache[:, resolved_local_indices], dtype=cupy.float32)
            else:
                if self.raw is None:
                    raise RuntimeError("GPU subset materialization requires raw backing storage or a dense cache.")
                gpu_subset = cupy.empty((self.shape[0], resolved_local_indices.shape[0]), dtype=cupy.float32, order="F")
                selected_variant_indices = self.variant_indices[resolved_local_indices]
                subset_batch_size = (
                    auto_batch_size_i8(self.shape[0])
                    if _supports_int8_batches(self.raw)
                    else auto_batch_size(self.shape[0])
                )
                if _supports_int8_batches(self.raw):
                    _upload_standardized_int8_tiles_overlapped(
                        cupy=cupy,
                        raw_int8=self.raw,
                        variant_indices=selected_variant_indices,
                        means=self.means,
                        scales=self.scales,
                        gpu_destination=gpu_subset,
                        sample_count=int(self.shape[0]),
                        upload_batch_size=int(subset_batch_size),
                        standardized_dtype=cupy.float32,
                    )
                else:
                    for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                        self.raw,
                        selected_variant_indices,
                        self.means,
                        self.scales,
                        batch_size=subset_batch_size,
                        cupy=cupy,
                    ):
                        gpu_subset[:, batch_slice] = standardized_batch
            cupy.cuda.Device().synchronize()
            self._cupy_subset_cache = gpu_subset
            self._cupy_subset_cache_local_indices = resolved_local_indices.copy()
            log(
                "    CuPy GPU subset ready "
                + f"({resolved_local_indices.shape[0]} variants, {nbytes / 1e9:.1f} GB)  mem={mem()}"
            )
            return self._cupy_subset_cache
        except (MemoryError, OSError, RuntimeError) as exc:
            self._cupy_subset_cache = None
            self._cupy_subset_cache_local_indices = None
            if not _is_cupy_out_of_memory(exc):
                raise
            _release_cupy_cached_memory(cupy)
            log(f"    CuPy GPU subset upload failed ({exc})  mem={mem()}")
            _warn_gpu_materialization_unavailable(f"subset allocation failed: {exc}", subset_plan)
            return None

    def try_materialize(self) -> bool:
        """Materialize into RAM if below the auto-materialize threshold.

        Returns True if now cached in memory, False if still streaming from disk.
        """
        if self._cupy_cache is not None or self._dense_cache is not None:
            return True
        nbytes = self.dense_bytes()
        if nbytes > MATERIALIZE_THRESHOLD_BYTES:
            return False
        log(f"    auto-materializing {self.shape[1]} variants x {self.shape[0]} samples ({nbytes / 1e9:.1f} GB) into RAM  mem={mem()}")
        self._dense_cache = self.materialize()
        self._jax_cache = None
        log(f"    materialized  mem={mem()}")
        return True

    def supports_jax_dense_ops(self) -> bool:
        return jax_dense_linear_algebra_preferred() and (
            self._dense_cache is not None
            or self._jax_cache is not None
            or _dense_array_cache_available(self._cupy_cache)
        )

    def _ensure_jax_cache(self) -> jax.Array:
        if self._jax_cache is not None:
            return self._jax_cache
        if self._dense_cache is not None:
            cache_source = self._dense_cache
            jax_cache = jnp.asarray(cache_source, dtype=gpu_compute_jax_dtype())
        elif _dense_array_cache_available(self._cupy_cache):
            cupy_cache_source = self._cupy_cache
            if hasattr(cupy_cache_source, "__dlpack__"):
                jax_cache = jax_dlpack.from_dlpack(cupy_cache_source).astype(gpu_compute_jax_dtype())
            else:
                jax_cache = jnp.asarray(np.asarray(cupy_cache_source), dtype=gpu_compute_jax_dtype())
        else:
            raise RuntimeError("JAX cache requires a dense materialized genotype matrix.")
        if isinstance(jax_cache, jax_core.Tracer):
            return jax_cache
        self._jax_cache = jax_cache
        return jax_cache

    def try_cache_locally(self) -> bool:
        """Rebase onto a local int8 memmap to avoid repeated upstream streaming passes."""
        if self.raw is None or self._dense_cache is not None or self._cupy_cache is not None:
            return False
        if not _supports_int8_batches(self.raw):
            return False
        if isinstance(self.raw, Int8RawGenotypeMatrix):
            # Already mmap-backed int8 (e.g. swapped in by io.py from a
            # persisted PLINK int8 cache). Hint the kernel to keep pages
            # resident so per-block GPU uploads run at RAM speed.
            _madvise_willneed_array(self.raw.matrix)
            return True
        batch_size = auto_batch_size_i8(self.shape[0])
        selected_variant_count = int(self.variant_indices.shape[0])
        log(
            "    caching reduced raw genotypes locally as int8 "
            + f"({selected_variant_count} variants x {self.shape[0]} samples)  mem={mem()}"
        )
        cache_directory = tempfile.TemporaryDirectory(prefix="svpgs-genotype-")
        cache_path = Path(cache_directory.name) / "reduced_raw_i8.npy"
        try:
            has_space, required_bytes, available_bytes = _has_sufficient_free_space_for_int8_npy(
                Path(cache_directory.name),
                self.shape,
                fortran_order=True,
            )
            if not has_space:
                log(
                    "    local int8 cache skipped: insufficient free space "
                    + f"(need~{required_bytes / 1e9:.1f} GB, free~{available_bytes / 1e9:.1f} GB)  mem={mem()}"
                )
                cache_directory.cleanup()
                return False
            raw_int8 = self.raw
            def _column_batches() -> Iterator[NDArray]:
                for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=batch_size):
                    yield raw_batch.values
            _stream_write_int8_npy(
                cache_path,
                shape=self.shape,
                column_batches=_column_batches(),
                fortran_order=True,
            )
            cache_mmap = np.load(cache_path, mmap_mode="r")
            _madvise_willneed_array(cache_mmap)
            rebased_raw = Int8RawGenotypeMatrix(cache_mmap)
            self.raw = rebased_raw
            self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
            self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
            if self.support_counts is not None:
                self.support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
            self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
            self.clear_sample_space_nystrom_cache()
            self._local_cache_directory = cache_directory
            self._cupy_subset_cache = None
            self._cupy_subset_cache_local_indices = None
            self._enable_hybrid_backend = False  # GPU streaming handles everything
            # Clear any pre-existing hybrid backend so _uses_hybrid_backend()
            # returns False and _streaming_gpu_context() can take the GPU path.
            self._sparse_backend = None
            self._dense_backend = None
            self._sparse_local_lookup = None
            self._dense_local_lookup = None
            log("    local int8 cache ready  mem=" + mem())
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            cache_directory.cleanup()
            log(f"    local int8 cache failed ({exc})  mem={mem()}")
            return False

    def try_cache_persistently(self, cache_path: Path) -> bool:
        """Persist a reduced raw int8 cache to a stable path for reuse across runs."""
        if self.raw is None:
            return False
        if not _supports_int8_batches(self.raw):
            return False
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        selected_variant_count = int(self.variant_indices.shape[0])
        # Short-circuit if a previous run already wrote this exact cache. Just rebase
        # onto the existing F-order int8 memmap; no rewrite, no double-build.
        if cache_path.exists():
            try:
                existing_mmap = np.load(cache_path, mmap_mode="r")
                if existing_mmap.shape == self.shape and existing_mmap.dtype == np.int8:
                    _madvise_willneed_array(existing_mmap)
                    self.raw = Int8RawGenotypeMatrix(existing_mmap)
                    self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
                    self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
                    if self.support_counts is not None:
                        self.support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
                    self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
                    self.clear_sample_space_nystrom_cache()
                    self._local_cache_directory = None
                    self._cupy_subset_cache = None
                    self._cupy_subset_cache_local_indices = None
                    self._enable_hybrid_backend = False
                    # Clear stale hybrid backend so GPU streaming is reachable.
                    self._sparse_backend = None
                    self._dense_backend = None
                    self._sparse_local_lookup = None
                    self._dense_local_lookup = None
                    log(f"    persistent int8 cache reused from {cache_path}  mem={mem()}")
                    return True
            except (OSError, ValueError) as exc:
                log(f"    existing persistent int8 cache unreadable ({exc}); rebuilding")
        batch_size = auto_batch_size_i8(self.shape[0])
        # Resumable write: a fixed sibling .partial file (NOT a random temp
        # dir) so a kill mid-write is recoverable on next launch.
        # ``_stream_write_int8_npy`` writes a sidecar ``.progress`` after
        # every batch; ``_scan_int8_npy_resume_point`` reads it back to find
        # the highest safely-written variant count.
        partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")
        resume_from_variants = _scan_int8_npy_resume_point(
            partial_path, shape=self.shape, fortran_order=True
        )
        if resume_from_variants > 0:
            log(
                f"    persistent int8 cache RESUMING from variant {resume_from_variants:,}"
                f"/{selected_variant_count:,} (sidecar progress file found)  mem={mem()}"
            )
        else:
            log(
                "    persisting reduced raw genotypes as int8 "
                + f"({selected_variant_count} variants x {self.shape[0]} samples) → {cache_path}  mem={mem()}"
            )
            # If a stale .partial exists but the sidecar didn't validate,
            # nuke it so we start fresh with a known-good header.
            if partial_path.exists():
                try:
                    partial_path.unlink()
                except OSError:
                    pass
                stale_progress = partial_path.with_suffix(partial_path.suffix + ".progress")
                try:
                    stale_progress.unlink(missing_ok=True)
                except OSError:
                    pass
        try:
            has_space, required_bytes, available_bytes = _has_sufficient_free_space_for_int8_npy(
                cache_path.parent,
                self.shape,
                fortran_order=True,
            )
            if not has_space:
                log(
                    "    persistent int8 cache skipped: insufficient free space "
                    + f"(need~{required_bytes / 1e9:.1f} GB, free~{available_bytes / 1e9:.1f} GB)  mem={mem()}"
                )
                return False
            raw_int8 = self.raw
            # Skip the first ``resume_from_variants`` of the variant index
            # array so we don't re-read+re-decode what's already on disk.
            # The iterator-side skip is critical — without it we'd pay the
            # full PLINK read cost again even though the .partial file
            # already has those bytes.
            variants_to_stream = (
                self.variant_indices[resume_from_variants:]
                if resume_from_variants > 0
                else self.variant_indices
            )

            def _column_batches() -> Iterator[NDArray]:
                for raw_batch in raw_int8.iter_column_batches_i8(variants_to_stream, batch_size=batch_size):
                    yield raw_batch.values

            _stream_write_int8_npy(
                partial_path,
                shape=self.shape,
                column_batches=_column_batches(),
                fortran_order=True,
                resume_from_variants=resume_from_variants,
            )
            partial_path.replace(cache_path)
            persisted_mmap = np.load(cache_path, mmap_mode="r")
            _madvise_willneed_array(persisted_mmap)
            persisted_raw = Int8RawGenotypeMatrix(persisted_mmap)
            self.raw = persisted_raw
            self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
            self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
            if self.support_counts is not None:
                self.support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
            self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
            self.clear_sample_space_nystrom_cache()
            self._local_cache_directory = None
            self._cupy_subset_cache = None
            self._cupy_subset_cache_local_indices = None
            # Don't rebuild hybrid backend — GPU streaming handles everything
            self._enable_hybrid_backend = False
            # Clear any pre-existing hybrid backend so _uses_hybrid_backend()
            # returns False and _streaming_gpu_context() can take the GPU path.
            self._sparse_backend = None
            self._dense_backend = None
            self._sparse_local_lookup = None
            self._dense_local_lookup = None
            log("    persistent int8 cache ready  mem=" + mem())
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            # IMPORTANT: do NOT delete partial_path here — a transient error
            # (disk pressure, sigterm, network blip) should leave the partial
            # bytes in place so the next run can resume rather than restart.
            log(f"    persistent int8 cache failed ({exc}); .partial retained for next-run resume  mem={mem()}")
            return False

    def subset(self, local_variant_indices: Sequence[int] | NDArray) -> StandardizedGenotypeMatrix:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        preserve_hybrid = (
            self._uses_hybrid_backend()
            and self._dense_cache is None
            and self._cupy_cache is None
            and self._jax_cache is None
        )
        subset = StandardizedGenotypeMatrix(
            raw=self.raw,
            means=self.means,
            scales=self.scales,
            variant_indices=self.variant_indices[resolved_local_indices],
            support_counts=self.support_counts,
            sample_count=self.shape[0],
            _enable_hybrid_backend=False if preserve_hybrid else self._enable_hybrid_backend,
        )
        if preserve_hybrid:
            sparse_mask, sparse_child_local_indices, dense_mask, dense_child_local_indices = self._hybrid_local_components(
                resolved_local_indices
            )
            if np.any(sparse_mask):
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                subset._sparse_backend = self._sparse_backend.subset(sparse_child_local_indices)
                subset._sparse_local_lookup = np.full(resolved_local_indices.shape[0], -1, dtype=np.int32)
                subset._sparse_local_lookup[sparse_mask] = np.arange(sparse_child_local_indices.shape[0], dtype=np.int32)
            if np.any(dense_mask):
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                subset._dense_backend = self._dense_backend.subset(dense_child_local_indices)
                subset._dense_local_lookup = np.full(resolved_local_indices.shape[0], -1, dtype=np.int32)
                subset._dense_local_lookup[dense_mask] = np.arange(dense_child_local_indices.shape[0], dtype=np.int32)
        if self._cupy_cache is not None:
            subset._cupy_cache = _cupy_cache_subset_columns(self._cupy_cache, resolved_local_indices)
            # Parent-lifetime invariant: ``_cupy_cache_subset_columns`` returns a
            # view/slice into ``self._cupy_cache``'s GPU buffer (no copy). If
            # ``self`` is garbage-collected before ``subset`` is consumed, the
            # underlying GPU memory is freed and ``subset._cupy_cache`` becomes a
            # dangling pointer. Retain a strong reference to ``self`` so the
            # parent's buffer outlives the subset.
            subset._parent_genotype_matrix = self
        if self._jax_cache is not None:
            subset._jax_cache = self._jax_cache[:, resolved_local_indices]
        elif self._dense_cache is not None:
            subset._dense_cache = np.asarray(self._dense_cache[:, resolved_local_indices], dtype=np.float32)
        subset._local_cache_directory = self._local_cache_directory
        return subset

    def iter_column_batches(
        self,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        if self._dense_cache is not None:
            safe_batch_size = max(int(batch_size), 1)
            for start_index in range(0, self._dense_cache.shape[1], safe_batch_size):
                local_indices = np.arange(start_index, min(start_index + safe_batch_size, self._dense_cache.shape[1]), dtype=np.int32)
                yield RawGenotypeBatch(
                    variant_indices=local_indices,
                    values=np.asarray(self._dense_cache[:, local_indices], dtype=np.float32),
                )
            return
        if self._uses_hybrid_backend():
            safe_batch_size = _effective_standardized_streaming_batch_size(
                self.shape[0],
                max(int(batch_size), 1),
                target_batch_bytes=STANDARDIZED_STREAMING_TARGET_BATCH_BYTES,
            )
            for start_index in range(0, self.shape[1], safe_batch_size):
                local_indices = np.arange(start_index, min(start_index + safe_batch_size, self.shape[1]), dtype=np.int32)
                yield RawGenotypeBatch(
                    variant_indices=local_indices,
                    values=self._materialize_hybrid_columns(local_indices, batch_size=safe_batch_size),
                )
            return
        if self.raw is None:
            raise RuntimeError("streaming genotype batches require raw backing storage or a materialized cache.")
        target_batch_bytes = (
            LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
            if _supports_int8_batches(self.raw)
            else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        )
        safe_batch_size = _effective_standardized_streaming_batch_size(
            self.shape[0],
            batch_size,
            target_batch_bytes=target_batch_bytes,
        )
        local_start = 0
        if _supports_int8_batches(self.raw):
            raw_int8 = self.raw
            for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=safe_batch_size):
                local_stop = local_start + raw_batch.variant_indices.shape[0]
                local_indices = np.arange(local_start, local_stop, dtype=np.int32)
                yield RawGenotypeBatch(
                    variant_indices=local_indices,
                    values=_standardize_batch_i8(
                        raw_batch.values,
                        self.means[raw_batch.variant_indices],
                        self.scales[raw_batch.variant_indices],
                    ),
                )
                local_start = local_stop
            return
        for raw_batch in self.raw.iter_column_batches(self.variant_indices, batch_size=safe_batch_size):
            local_stop = local_start + raw_batch.variant_indices.shape[0]
            local_indices = np.arange(local_start, local_stop, dtype=np.int32)
            yield RawGenotypeBatch(
                variant_indices=local_indices,
                values=_standardize_batch(
                    raw_batch.values,
                    self.means[raw_batch.variant_indices],
                    self.scales[raw_batch.variant_indices],
                ),
            )
            local_start = local_stop

    def materialize(self, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        if self._dense_cache is not None:
            return self._dense_cache
        if self._cupy_cache is not None:
            if _cupy_cache_is_int8_standardized(self._cupy_cache) or _cupy_cache_is_sharded(self._cupy_cache):
                dense_from_cupy = np.asarray(self._cupy_cache, dtype=np.float32)
            else:
                dense_from_cupy = self._cupy_cache.get()  # cupy -> numpy
            self._dense_cache = dense_from_cupy
            self._jax_cache = None
            return dense_from_cupy
        matrix = np.empty(self.shape, dtype=np.float32)
        for batch in self.iter_column_batches(batch_size=batch_size):
            matrix[:, batch.variant_indices] = batch.values
        self._dense_cache = matrix
        self._jax_cache = None
        return matrix

    def _streaming_gpu_context(
        self, batch_size: int, *, cupy: Any = None, dtype: Any = None
    ) -> tuple[Any, int | None]:
        if self._cupy_cache is not None or self.supports_jax_dense_ops() or self.raw is None or self._uses_hybrid_backend():
            return None, None
        resolved_cupy = _try_import_cupy() if cupy is None else cupy
        if resolved_cupy is None:
            return None, None
        return resolved_cupy, _gpu_streaming_batch_size(
            self.raw,
            sample_count=self.shape[0],
            requested_batch_size=batch_size,
            cupy=resolved_cupy,
            dtype=dtype,
        )

    def _gpu_variant_matmul(
        self,
        matrix: Any,
        *,
        batch_size: int,
        cupy: Any,
        dtype: Any = None,
        local_variant_indices: NDArray | None = None,
        progress_label: str | None = None,
    ) -> Any:
        resolved_dtype = _cupy_compute_dtype(cupy) if dtype is None else dtype
        matrix_gpu = cupy.asarray(matrix, dtype=resolved_dtype)
        if matrix_gpu.ndim == 1:
            matrix_gpu = matrix_gpu[:, None]
            vector_input = True
        elif matrix_gpu.ndim == 2:
            vector_input = False
        else:
            raise ValueError("GPU genotype matmul expects a vector or matrix right-hand side.")
        any_fn = getattr(cupy, "any", np.any)
        if not bool(any_fn(matrix_gpu)):
            result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
            return result_gpu[:, 0] if vector_input else result_gpu
        resolved_local_indices = (
            np.arange(self.shape[1], dtype=np.int32)
            if local_variant_indices is None
            else np.asarray(local_variant_indices, dtype=np.int32)
        )
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if matrix_gpu.shape[0] != resolved_local_indices.shape[0]:
            raise ValueError("GPU genotype matmul right-hand side row count must match the selected variant count.")
        if resolved_local_indices.size == 0:
            result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
            return result_gpu[:, 0] if vector_input else result_gpu
        if self._cupy_cache is not None:
            if local_variant_indices is None:
                cache = self._cupy_cache
                if _cupy_cache_is_sharded(cache):
                    result_gpu = _gpu_sharded_cache_variant_matmul(
                        cache,
                        matrix_gpu,
                        local_variant_indices=None,
                        cupy=cupy,
                        dtype=resolved_dtype,
                    )
                else:
                    result_gpu = _gpu_single_cache_variant_matmul(
                        cache,
                        matrix_gpu,
                        local_variant_indices=None,
                        cupy=cupy,
                        dtype=resolved_dtype,
                    )
            else:
                cache = self._cupy_cache
                if _cupy_cache_is_sharded(cache):
                    result_gpu = _gpu_sharded_cache_variant_matmul(
                        cache,
                        matrix_gpu,
                        local_variant_indices=resolved_local_indices,
                        cupy=cupy,
                        dtype=resolved_dtype,
                    )
                elif _cupy_cache_is_int8_standardized(cache) or _cupy_cache_is_fp16_resident(cache):
                    result_gpu = _gpu_single_cache_variant_matmul(
                        cache,
                        matrix_gpu,
                        local_variant_indices=resolved_local_indices,
                        cupy=cupy,
                        dtype=resolved_dtype,
                    )
                else:
                    result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
                    for operand_slice, standardized_batch in _iter_selected_cupy_cache_standardized_batches(
                        cache,
                        resolved_local_indices,
                        sample_count=self.shape[0],
                        batch_size=batch_size,
                        cupy=cupy,
                        dtype=resolved_dtype,
                    ):
                        result_gpu += standardized_batch @ matrix_gpu[operand_slice, :]
            return result_gpu[:, 0] if vector_input else result_gpu
        streaming_cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        )
        if streaming_cupy is None or streaming_batch_size is None or self.raw is None:
            raise RuntimeError("GPU genotype matmul requires a GPU cache or a streaming raw backend.")
        active_variant_indices = self.variant_indices[resolved_local_indices]
        result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
        raw_matrix = self.raw
        total_variants = int(resolved_local_indices.shape[0])
        completed_variants = 0
        last_logged_variants = 0
        log_interval = max(total_variants // 50, 1)
        import time as _time
        _t_start = _time.monotonic()
        if progress_label is not None:
            log(f"        {progress_label}: start streaming {total_variants:,} variants (batch_size={streaming_batch_size})  mem={mem()}")
        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
            raw_matrix,
            active_variant_indices,
            self.means,
            self.scales,
            batch_size=streaming_batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        ):
            result_gpu += standardized_batch @ matrix_gpu[batch_slice, :]
            if progress_label is not None:
                completed_variants = (
                    batch_slice.stop
                    if isinstance(batch_slice, slice)
                    else completed_variants + standardized_batch.shape[1]
                )
                if completed_variants - last_logged_variants >= log_interval:
                    last_logged_variants = completed_variants
                    _elapsed = _time.monotonic() - _t_start
                    _rate = completed_variants / max(_elapsed, 1e-6)
                    _eta = (total_variants - completed_variants) / max(_rate, 1e-6)
                    log(
                        f"        {progress_label}: {completed_variants:,}/{total_variants:,} "
                        f"({100*completed_variants/total_variants:.1f}%) "
                        f"elapsed={_elapsed:.0f}s rate={_rate:,.0f}v/s eta={_eta:.0f}s  mem={mem()}"
                    )
        if progress_label is not None:
            _elapsed = _time.monotonic() - _t_start
            log(f"        {progress_label}: done {total_variants:,} variants in {_elapsed:.1f}s  mem={mem()}")
        return result_gpu[:, 0] if vector_input else result_gpu

    def _gpu_transpose_matmul(
        self,
        matrix: Any,
        *,
        batch_size: int,
        cupy: Any,
        dtype: Any = None,
        progress_label: str | None = None,
    ) -> Any:
        resolved_dtype = _cupy_compute_dtype(cupy) if dtype is None else dtype
        matrix_gpu = cupy.asarray(matrix, dtype=resolved_dtype)
        if matrix_gpu.ndim == 1:
            matrix_gpu = matrix_gpu[:, None]
            vector_input = True
        elif matrix_gpu.ndim == 2:
            vector_input = False
        else:
            raise ValueError("GPU genotype transpose matmul expects a vector or matrix right-hand side.")
        if matrix_gpu.shape[0] != self.shape[0]:
            raise ValueError("GPU genotype transpose matmul right-hand side row count must match the sample count.")
        if self._cupy_cache is not None:
            if _cupy_cache_is_sharded(self._cupy_cache):
                result_gpu = _gpu_sharded_cache_transpose_matmul(
                    self._cupy_cache,
                    matrix_gpu,
                    cupy=cupy,
                    dtype=resolved_dtype,
                )
            else:
                result_gpu = _gpu_single_cache_transpose_matmul(
                    self._cupy_cache,
                    matrix_gpu,
                    cupy=cupy,
                    dtype=resolved_dtype,
                )
            return result_gpu[:, 0] if vector_input else result_gpu
        streaming_cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        )
        if streaming_cupy is None or streaming_batch_size is None or self.raw is None:
            raise RuntimeError("GPU genotype transpose matmul requires a GPU cache or a streaming raw backend.")
        raw_matrix = self.raw
        if _supports_int8_batches(raw_matrix):
            result_gpu = _gpu_int8_transpose_matmul(
                raw_int8=raw_matrix,
                variant_indices=self.variant_indices,
                means=self.means,
                scales=self.scales,
                matrix_gpu=matrix_gpu,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=resolved_dtype,
                progress_label=progress_label,
            )
            return result_gpu[:, 0] if vector_input else result_gpu
        result_gpu = cupy.empty((self.shape[1], matrix_gpu.shape[1]), dtype=resolved_dtype)
        total_variants = self.shape[1]
        completed_variants = 0
        last_logged_variants = 0
        log_interval = max(total_variants // 50, 1)
        import time as _time
        _t_start = _time.monotonic()
        if progress_label is not None:
            log(f"        {progress_label}: start streaming {total_variants:,} variants (batch_size={streaming_batch_size})  mem={mem()}")
        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
            raw_matrix,
            self.variant_indices,
            self.means,
            self.scales,
            batch_size=streaming_batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        ):
            result_gpu[batch_slice, :] = standardized_batch.T @ matrix_gpu
            if progress_label is not None:
                completed_variants = (
                    batch_slice.stop
                    if isinstance(batch_slice, slice)
                    else completed_variants + standardized_batch.shape[1]
                )
                if completed_variants - last_logged_variants >= log_interval:
                    last_logged_variants = completed_variants
                    _elapsed = _time.monotonic() - _t_start
                    _rate = completed_variants / max(_elapsed, 1e-6)
                    _eta = (total_variants - completed_variants) / max(_rate, 1e-6)
                    log(
                        f"        {progress_label}: {completed_variants:,}/{total_variants:,} "
                        f"({100*completed_variants/total_variants:.1f}%) "
                        f"elapsed={_elapsed:.0f}s rate={_rate:,.0f}v/s eta={_eta:.0f}s  mem={mem()}"
                    )
        if progress_label is not None:
            _elapsed = _time.monotonic() - _t_start
            log(f"        {progress_label}: done {total_variants:,} variants in {_elapsed:.1f}s  mem={mem()}")
        return result_gpu[:, 0] if vector_input else result_gpu

    def gpu_matmat(
        self,
        matrix: Any,
        *,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        cupy: Any = None,
        dtype: Any = None,
    ) -> Any:
        resolved_cupy = _try_import_cupy() if cupy is None else cupy
        if resolved_cupy is None:
            raise RuntimeError("GPU genotype matmul requires CuPy.")
        return self._gpu_variant_matmul(
            matrix,
            batch_size=batch_size,
            cupy=resolved_cupy,
            dtype=dtype,
        )

    def gpu_transpose_matmat(
        self,
        matrix: Any,
        *,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        cupy: Any = None,
        dtype: Any = None,
    ) -> Any:
        resolved_cupy = _try_import_cupy() if cupy is None else cupy
        if resolved_cupy is None:
            raise RuntimeError("GPU genotype transpose matmul requires CuPy.")
        return self._gpu_transpose_matmul(
            matrix,
            batch_size=batch_size,
            cupy=resolved_cupy,
            dtype=dtype,
        )

    def _iter_selected_column_batches(
        self,
        local_variant_indices: NDArray,
        *,
        batch_size: int,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved_local_indices.size == 0:
            return
        if self._dense_cache is not None:
            safe_batch_size = max(int(batch_size), 1)
            for start_index in range(0, resolved_local_indices.shape[0], safe_batch_size):
                batch_local_indices = resolved_local_indices[start_index : start_index + safe_batch_size]
                yield RawGenotypeBatch(
                    variant_indices=batch_local_indices,
                    values=np.asarray(self._dense_cache[:, batch_local_indices], dtype=np.float32),
                )
            return
        if self._uses_hybrid_backend():
            safe_batch_size = _effective_standardized_streaming_batch_size(
                self.shape[0],
                max(int(batch_size), 1),
                target_batch_bytes=STANDARDIZED_STREAMING_TARGET_BATCH_BYTES,
            )
            for start_index in range(0, resolved_local_indices.shape[0], safe_batch_size):
                batch_local_indices = resolved_local_indices[start_index : start_index + safe_batch_size]
                yield RawGenotypeBatch(
                    variant_indices=batch_local_indices,
                    values=self._materialize_hybrid_columns(batch_local_indices, batch_size=safe_batch_size),
                )
            return
        if self.raw is None:
            raise RuntimeError("streaming genotype batches require raw backing storage or a materialized cache.")
        target_batch_bytes = (
            LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
            if _supports_int8_batches(self.raw)
            else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        )
        safe_batch_size = _effective_standardized_streaming_batch_size(
            self.shape[0],
            batch_size,
            target_batch_bytes=target_batch_bytes,
        )
        selected_variant_indices = self.variant_indices[resolved_local_indices]
        selected_means = np.asarray(self.means[selected_variant_indices], dtype=np.float32)
        selected_scales = np.asarray(self.scales[selected_variant_indices], dtype=np.float32)
        local_start = 0
        if _supports_int8_batches(self.raw):
            raw_int8 = self.raw
            for raw_batch in raw_int8.iter_column_batches_i8(selected_variant_indices, batch_size=safe_batch_size):
                batch_width = raw_batch.variant_indices.shape[0]
                local_stop = local_start + batch_width
                batch_slice = slice(local_start, local_stop)
                yield RawGenotypeBatch(
                    variant_indices=resolved_local_indices[batch_slice],
                    values=_standardize_batch_i8(
                        raw_batch.values,
                        selected_means[batch_slice],
                        selected_scales[batch_slice],
                    ),
                )
                local_start = local_stop
            return
        for raw_batch in self.raw.iter_column_batches(selected_variant_indices, batch_size=safe_batch_size):
            batch_width = raw_batch.variant_indices.shape[0]
            local_stop = local_start + batch_width
            batch_slice = slice(local_start, local_stop)
            yield RawGenotypeBatch(
                variant_indices=resolved_local_indices[batch_slice],
                values=_standardize_batch(
                    raw_batch.values,
                    selected_means[batch_slice],
                    selected_scales[batch_slice],
                ),
            )
            local_start = local_stop

    def matvec_numpy(self, coefficients: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        coeff_np = _normalize_numpy_vector_operand(
            coefficients,
            expected_length=self.shape[1],
            shape_error="coefficient vector must match genotype column count.",
            finite_name="coefficient vector",
        )
        # Bitpacked-device fast path: only valid when this SGM exposes the
        # ENTIRE underlying bitpacked matrix (variant_indices is identity).
        # The stochastic-block path constructs subset SGMs whose
        # ``variant_indices`` selects a proper subset of the parent's
        # 91084 columns; bp.matvec_numpy doesn't know about that and would
        # return a (n_samples,) result for the *full* parent — but the
        # caller expects a result consistent with subset columns. Detect
        # the identity case and route only then.
        raw = getattr(self, "raw", None)
        if (
            raw is not None
            and getattr(raw, "_packed", None) is not None
            and self.shape[1] == int(getattr(raw, "n_variants", -1))
        ):
            if _sgm_variant_indices_is_identity(self):
                try:
                    result = raw.matvec_numpy(coeff_np)
                    return np.asarray(result, dtype=coeff_np.dtype)
                except Exception as _bp_exc:  # noqa: BLE001 - fall back rather than abort fit
                    _warn_bitpacked_dispatch_fallback("matvec_numpy", _bp_exc)
        active_local_indices = _active_vector_local_indices(coeff_np)
        if active_local_indices.size == 0:
            return np.zeros(self.shape[0], dtype=coeff_np.dtype)
        selected_local_indices = _selected_or_all_local_indices(active_local_indices, self.shape[1])
        selected_coefficients = coeff_np if selected_local_indices is None else coeff_np[active_local_indices]
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache) or _cupy_cache_is_sharded(self._cupy_cache):
                    raise RuntimeError("CuPy cache requires CuPy runtime.")
                dense_cache = (
                    np.asarray(self._cupy_cache, dtype=coeff_np.dtype)
                    if selected_local_indices is None
                    else np.asarray(_cupy_cache_subset_columns(self._cupy_cache, selected_local_indices), dtype=coeff_np.dtype)
                )
                return np.asarray(dense_cache @ selected_coefficients, dtype=coeff_np.dtype)
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_coefficients,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                    local_variant_indices=selected_local_indices,
                ),
                dtype=coeff_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_coefficients,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                    local_variant_indices=selected_local_indices,
                    progress_label="GPU matvec",
                ),
                dtype=coeff_np.dtype,
            )
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            result = np.zeros(self.shape[0], dtype=coeff_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                result += self._sparse_backend.matvec(coeff_np[sparse_parent_local_indices])
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                result += self._dense_backend.matvec_numpy(coeff_np[dense_parent_local_indices], batch_size=batch_size)
            return np.asarray(result, dtype=coeff_np.dtype)
        result = np.zeros(self.shape[0], dtype=coeff_np.dtype)
        _total_variants = int(active_local_indices.shape[0])
        _completed_variants = 0
        _last_log_variants = 0
        _log_interval = max(_total_variants // 10, 1)
        for batch in self._iter_selected_column_batches(active_local_indices, batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=result.dtype)
            result += batch_values @ coeff_np[batch.variant_indices]
            _completed_variants += len(batch.variant_indices)
            if _completed_variants - _last_log_variants >= _log_interval:
                _last_log_variants = _completed_variants
                log(f"        matvec: {_completed_variants:,}/{_total_variants:,} ({100*_completed_variants/_total_variants:.0f}%)  mem={mem()}")
        return result

    def matvec_jax(self, coefficients: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            coeff_jax = _normalize_jax_vector_operand(
                coefficients,
                expected_length=self.shape[1],
                shape_error="coefficient vector must match genotype column count.",
                finite_name="coefficient vector",
            )
            return self._ensure_jax_cache() @ coeff_jax
        return _as_gpu_compute_jax(self.matvec_numpy(coefficients, batch_size=batch_size))

    def matvec(self, coefficients: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std @ beta with JAX as the public return type."""
        return self.matvec_jax(coefficients, batch_size=batch_size)

    def matmat_numpy(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        m_np = _normalize_numpy_matrix_operand(
            matrix,
            expected_rows=self.shape[1],
            shape_error="variant matrix must match genotype column count.",
            finite_name="variant matrix",
        )
        active_local_indices = _active_matrix_row_local_indices(m_np)
        if active_local_indices.size == 0:
            return np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
        selected_local_indices = _selected_or_all_local_indices(active_local_indices, self.shape[1])
        selected_matrix = m_np if selected_local_indices is None else m_np[active_local_indices, :]
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache) or _cupy_cache_is_sharded(self._cupy_cache):
                    raise RuntimeError("CuPy cache requires CuPy runtime.")
                dense_cache = (
                    np.asarray(self._cupy_cache, dtype=m_np.dtype)
                    if selected_local_indices is None
                    else np.asarray(_cupy_cache_subset_columns(self._cupy_cache, selected_local_indices), dtype=m_np.dtype)
                )
                return np.asarray(dense_cache @ selected_matrix, dtype=m_np.dtype)
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_matrix,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                    local_variant_indices=selected_local_indices,
                ),
                dtype=m_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_matrix,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                    local_variant_indices=selected_local_indices,
                ),
                dtype=m_np.dtype,
            )
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            output = np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                output += self._sparse_backend.matmat(m_np[sparse_parent_local_indices, :])
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                output += self._dense_backend.matmat_numpy(m_np[dense_parent_local_indices, :], batch_size=batch_size)
            return output
        output = np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
        for batch in self._iter_selected_column_batches(active_local_indices, batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=output.dtype)
            output += batch_values @ m_np[batch.variant_indices, :]
        return output

    def matmat_jax(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            matrix_jax = _normalize_jax_matrix_operand(
                matrix,
                expected_rows=self.shape[1],
                shape_error="variant matrix must match genotype column count.",
                finite_name="variant matrix",
            )
            return self._ensure_jax_cache() @ matrix_jax
        return _as_gpu_compute_jax(self.matmat_numpy(matrix, batch_size=batch_size))

    def matmat(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std @ M with JAX as the public return type."""
        return self.matmat_jax(matrix, batch_size=batch_size)

    def transpose_matvec_numpy(self, vector: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        v_np = _normalize_numpy_vector_operand(
            vector,
            expected_length=self.shape[0],
            shape_error="sample vector must match genotype row count.",
            finite_name="sample vector",
        )
        if not np.any(v_np):
            return np.zeros(self.shape[1], dtype=v_np.dtype)
        # Bitpacked-device fast path mirror of matvec_numpy: only valid when
        # this SGM exposes the entire underlying bitpacked matrix
        # (variant_indices is identity). Subset SGMs fall through to the
        # streaming path which handles column subsetting correctly.
        raw = getattr(self, "raw", None)
        if (
            raw is not None
            and getattr(raw, "_packed", None) is not None
            and self.shape[1] == int(getattr(raw, "n_variants", -1))
        ):
            if _sgm_variant_indices_is_identity(self):
                try:
                    result = raw.rmatvec_numpy(v_np)
                    return np.asarray(result, dtype=v_np.dtype)
                except Exception as _bp_exc:  # noqa: BLE001 - fall back rather than abort fit
                    _warn_bitpacked_dispatch_fallback("transpose_matvec_numpy", _bp_exc)
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache) or _cupy_cache_is_sharded(self._cupy_cache):
                    raise RuntimeError("CuPy cache requires CuPy runtime.")
                return np.asarray(np.asarray(self._cupy_cache, dtype=v_np.dtype).T @ v_np, dtype=v_np.dtype)
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    v_np,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                ),
                dtype=v_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    v_np,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                    progress_label="GPU transpose_matvec",
                ),
                dtype=v_np.dtype,
            )
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            output = np.empty(self.shape[1], dtype=v_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                output[sparse_parent_local_indices] = self._sparse_backend.transpose_matvec(v_np)
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                output[dense_parent_local_indices] = self._dense_backend.transpose_matvec_numpy(v_np, batch_size=batch_size)
            return output
        output = np.empty(self.shape[1], dtype=v_np.dtype)
        _total_variants = self.shape[1]
        _completed_variants = 0
        _last_log_variants = 0
        _log_interval = max(_total_variants // 10, 1)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices] = np.asarray(batch.values, dtype=output.dtype).T @ v_np
            _completed_variants += len(batch.variant_indices)
            if _completed_variants - _last_log_variants >= _log_interval:
                _last_log_variants = _completed_variants
                log(f"        transpose_matvec: {_completed_variants:,}/{_total_variants:,} ({100*_completed_variants/_total_variants:.0f}%)  mem={mem()}")
        return output

    def transpose_matvec_jax(self, vector: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            vector_jax = _normalize_jax_vector_operand(
                vector,
                expected_length=self.shape[0],
                shape_error="sample vector must match genotype row count.",
                finite_name="sample vector",
            )
            return self._ensure_jax_cache().T @ vector_jax
        return _as_gpu_compute_jax(self.transpose_matvec_numpy(vector, batch_size=batch_size))

    def transpose_matvec(self, vector: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std^T @ v with JAX as the public return type."""
        return self.transpose_matvec_jax(vector, batch_size=batch_size)

    def transpose_matmat_numpy(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        m_np = _normalize_numpy_matrix_operand(
            matrix,
            expected_rows=self.shape[0],
            shape_error="sample matrix must match genotype row count.",
            finite_name="sample matrix",
        )
        if not np.any(m_np):
            return np.zeros((self.shape[1], m_np.shape[1]), dtype=m_np.dtype)
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache) or _cupy_cache_is_sharded(self._cupy_cache):
                    raise RuntimeError("CuPy cache requires CuPy runtime.")
                return np.asarray(np.asarray(self._cupy_cache, dtype=m_np.dtype).T @ m_np, dtype=m_np.dtype)
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    m_np,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                ),
                dtype=m_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    m_np,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                ),
                dtype=m_np.dtype,
            )
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            output = np.empty((self.shape[1], m_np.shape[1]), dtype=m_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                output[sparse_parent_local_indices, :] = self._sparse_backend.transpose_matmat(m_np)
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                output[dense_parent_local_indices, :] = self._dense_backend.transpose_matmat_numpy(m_np, batch_size=batch_size)
            return output
        output = np.empty((self.shape[1], m_np.shape[1]), dtype=m_np.dtype)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices, :] = np.asarray(batch.values, dtype=output.dtype).T @ m_np
        return output

    def transpose_matmat_jax(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            matrix_jax = _normalize_jax_matrix_operand(
                matrix,
                expected_rows=self.shape[0],
                shape_error="sample matrix must match genotype row count.",
                finite_name="sample matrix",
            )
            return self._ensure_jax_cache().T @ matrix_jax
        return _as_gpu_compute_jax(self.transpose_matmat_numpy(matrix, batch_size=batch_size))

    def transpose_matmat(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std^T @ M with JAX as the public return type."""
        return self.transpose_matmat_jax(matrix, batch_size=batch_size)


def _resolve_variant_indices(
    variant_count: int,
    variant_indices: Sequence[int] | NDArray | None,
) -> NDArray:
    if variant_indices is None:
        return np.arange(variant_count, dtype=np.int32)
    resolved_indices = np.asarray(variant_indices, dtype=np.int32)
    if resolved_indices.ndim != 1:
        raise ValueError("variant_indices must be 1D.")
    return resolved_indices


# Auto-choose a good batch size based on sample count and memory budget.
# Larger batches = fewer I/O round-trips = faster, but each batch must fit
# in memory.  With 447k samples at 4 bytes each, one variant = ~1.7 MB,
# so 500 MB budget => ~279 variants per batch.
def auto_batch_size(sample_count: int) -> int:
    """Pick a genotype batch size that fits within the memory budget."""
    if sample_count < 1:
        return DEFAULT_GENOTYPE_BATCH_SIZE
    bytes_per_variant = sample_count * 4  # float32
    memory_capped = max(BED_READER_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), 1)
    return max(MIN_BED_READER_BATCH_SIZE, min(DEFAULT_GENOTYPE_BATCH_SIZE, memory_capped))


def auto_batch_size_i8(sample_count: int) -> int:
    """Pick an int8-native batch size that fits the int8 decode budget."""
    if sample_count < 1:
        return DEFAULT_GENOTYPE_BATCH_SIZE
    bytes_per_variant = sample_count  # int8
    memory_capped = max(PLINK_INT8_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), 1)
    return max(
        MIN_BED_READER_BATCH_SIZE,
        min(int(memory_capped), 16 * DEFAULT_GENOTYPE_BATCH_SIZE),
    )


def _effective_standardized_streaming_batch_size(
    sample_count: int,
    requested_batch_size: int,
    target_batch_bytes: int = STANDARDIZED_STREAMING_TARGET_BATCH_BYTES,
) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive.")
    if target_batch_bytes < 1:
        raise ValueError("target_batch_bytes must be positive.")
    bytes_per_variant = sample_count * np.dtype(np.float32).itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    return max(1, min(requested_batch_size, max(memory_capped_batch_size, MIN_BED_READER_BATCH_SIZE)))


# Cap the batch size so each batch doesn't exceed the memory budget.
# With 447k samples at 4 bytes (float32) each, one variant = ~1.7 MB.
# At the 500 MB budget that's ~279 variants per batch.  We also enforce
# a minimum of 32 variants per batch to avoid excessive I/O overhead.
def _effective_bed_reader_batch_size(
    sample_count: int,
    requested_batch_size: int,
) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive.")
    bytes_per_variant = sample_count * np.dtype(np.float32).itemsize
    memory_capped_batch_size = max(BED_READER_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), 1)
    return max(1, min(requested_batch_size, max(memory_capped_batch_size, MIN_BED_READER_BATCH_SIZE)))


# Optimization: if the requested variant indices are consecutive (e.g. [5,6,7,8]),
# convert to a slice (5:9) which the PLINK reader can read much faster (sequential disk I/O)
# than random-access indexing.  Falls back to an index array for non-contiguous indices.
def _contiguous_index_or_slice(indices: NDArray) -> slice | NDArray:
    resolved_indices = np.asarray(indices, dtype=np.intp)
    if resolved_indices.ndim != 1:
        raise ValueError("indices must be 1D.")
    if resolved_indices.size == 0:
        return slice(0, 0, 1)
    if resolved_indices.size == 1:
        start = int(resolved_indices[0])
        return slice(start, start + 1, 1)
    deltas = np.diff(resolved_indices)
    if np.all(deltas == 1):
        return slice(int(resolved_indices[0]), int(resolved_indices[-1]) + 1, 1)
    return np.ascontiguousarray(resolved_indices, dtype=np.intp)


def _read_int8_columns_one_shot(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    variant_indices: NDArray,
) -> NDArray | None:
    resolved_indices = np.asarray(variant_indices, dtype=np.int32)
    if resolved_indices.ndim != 1:
        raise ValueError("variant_indices must be 1D.")
    if resolved_indices.size == 0:
        return np.empty((raw.shape[0], 0), dtype=np.int8, order="F")
    if isinstance(raw, Int8RawGenotypeMatrix):
        column_index = _contiguous_index_or_slice(resolved_indices)
        return np.asfortranarray(raw.matrix[:, column_index], dtype=np.int8)
    if isinstance(raw, PlinkRawGenotypeMatrix):
        reader = raw._bed_reader()
        return np.asfortranarray(raw._read_batch_i8(reader, resolved_indices), dtype=np.int8)
    if isinstance(raw, IndexedRawGenotypeMatrix) and _supports_int8_batches(raw.child):
        child_block = _read_int8_columns_one_shot(
            raw.child,
            raw._child_columns(resolved_indices),
        )
        return None if child_block is None else np.asfortranarray(child_block, dtype=np.int8)
    if isinstance(raw, RowSubsetRawGenotypeMatrix) and _supports_int8_batches(raw.child):
        child_sample_count = max(int(raw.child.shape[0]), 1)
        subset_sample_count = max(int(raw.shape[0]), 1)
        if child_sample_count > int(subset_sample_count * ROW_SUBSET_ONE_SHOT_MAX_SAMPLE_RATIO):
            return None
        child_block = _read_int8_columns_one_shot(
            raw.child,
            resolved_indices,
        )
        if child_block is None:
            return None
        return np.asfortranarray(child_block[raw.row_indices, :], dtype=np.int8)
    if isinstance(raw, ConcatenatedRawGenotypeMatrix) and all(_supports_int8_batches(child) for child in raw.children):
        child_ids = np.searchsorted(raw._variant_offsets[1:], resolved_indices, side="right")
        result = np.empty((raw.shape[0], resolved_indices.shape[0]), dtype=np.int8, order="F")
        for child_index in np.unique(child_ids):
            child_positions = np.nonzero(child_ids == child_index)[0]
            child_variant_indices = resolved_indices[child_positions] - int(raw._variant_offsets[child_index])
            child_block = _read_int8_columns_one_shot(
                cast(Int8BatchCapable, raw.children[int(child_index)]),
                child_variant_indices,
            )
            if child_block is None:
                return None
            result[:, child_positions] = child_block
        return result
    return None


def _standardize_batch(batch: NDArray, means: NDArray, scales: NDArray) -> NDArray:
    batch_f32 = np.asarray(batch, dtype=np.float32)
    means_f32 = np.asarray(means, dtype=np.float32)
    scales_f32 = np.asarray(scales, dtype=np.float32)
    standardized = batch_f32 - means_f32[None, :]
    standardized[np.isnan(batch_f32)] = 0.0
    standardized /= scales_f32[None, :]
    return standardized.astype(np.float32, copy=False)


def _int8_batch_to_float32(batch: NDArray) -> NDArray:
    batch_i8 = np.asarray(batch, dtype=np.int8)
    batch_f32 = batch_i8.astype(np.float32)
    batch_f32[batch_i8 == PLINK_MISSING_INT8] = np.nan
    return batch_f32


def _standardize_batch_i8(batch: NDArray, means: NDArray, scales: NDArray) -> NDArray:
    batch_i8 = np.asarray(batch, dtype=np.int8)
    means_f32 = np.asarray(means, dtype=np.float32)
    scales_f32 = np.asarray(scales, dtype=np.float32)
    standardized = batch_i8.astype(np.float32)
    missing_mask = batch_i8 == PLINK_MISSING_INT8
    standardized -= means_f32[None, :]
    standardized[missing_mask] = 0.0
    standardized /= scales_f32[None, :]
    return standardized.astype(np.float32, copy=False)


def make_dense_raw_genotype_matrix(
    values: NDArray,
    properties: dict[str, Any] | None = None,
    *,
    prefer: str = "bitpacked",
) -> RawGenotypeMatrix:
    """Build a dense in-memory ``RawGenotypeMatrix`` from a host dosage array.

    When ``prefer == "bitpacked"`` and the bitpacked device backend is
    importable AND CuPy is present, the dosage matrix is encoded with the
    same PLINK 1.9 2-bit scheme as ``sv_pgs.plink._encode_variant`` and
    uploaded to GPU HBM as a :class:`BitpackedDeviceMatrix`. Otherwise we
    fall back to the host int8 ``Int8RawGenotypeMatrix``.

    Parameters
    ----------
    values:
        ``(n_samples, n_variants)`` host array of dosages (0/1/2/NaN) as
        float32, float64, or int8 (with ``PLINK_MISSING_INT8`` as missing).
    properties:
        Optional metadata dict to attach to the returned matrix.
    prefer:
        ``"bitpacked"`` (default) or ``"int8"``.
    """
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError("values must be 2D (n_samples, n_variants).")
    if arr.dtype == np.int8:
        host_i8 = arr
    else:
        host_f = np.asarray(arr, dtype=np.float32)
        host_i8 = np.where(np.isnan(host_f), np.float32(PLINK_MISSING_INT8), host_f).astype(np.int8)

    from sv_pgs.progress import log as _log  # local: avoid circular import at module load
    cupy_available = _try_import_cupy() is not None
    use_bitpacked = (
        prefer == "bitpacked"
        and BitpackedDeviceMatrix is not None
        and cupy_available
    )
    if not use_bitpacked:
        if prefer == "bitpacked":
            reason = (
                "BitpackedDeviceMatrix import failed"
                if BitpackedDeviceMatrix is None
                else "CuPy unavailable"
            )
            _log(
                f"make_dense_raw_genotype_matrix: bitpacked upgrade: SKIPPED "
                f"(reason: {reason})"
            )
        else:
            _log(
                f"make_dense_raw_genotype_matrix: bitpacked upgrade: SKIPPED "
                f"(reason: prefer={prefer!r} is not 'bitpacked')"
            )
        return Int8RawGenotypeMatrix(matrix=host_i8)

    from sv_pgs.plink import _encode_variant  # local import: heavy module
    cp = _try_import_cupy()
    assert cp is not None  # narrowed by use_bitpacked
    n_samples, n_variants = int(host_i8.shape[0]), int(host_i8.shape[1])
    bytes_per_variant = (n_samples + 3) // 4
    packed_host = np.empty((n_variants, bytes_per_variant), dtype=np.uint8)
    host_f = host_i8.astype(np.float32)
    host_f[host_i8 == PLINK_MISSING_INT8] = np.nan
    for v in range(n_variants):
        packed_host[v] = np.frombuffer(
            _encode_variant(host_f[:, v], bytes_per_variant=bytes_per_variant),
            dtype=np.uint8,
        )
    # Per-variant mean/std over non-missing samples (mean-impute std==0 → 1).
    mask = host_i8 != PLINK_MISSING_INT8
    counts = mask.sum(axis=0).astype(np.float64)
    safe_counts = np.where(counts > 0, counts, 1.0)
    sums = np.where(mask, host_i8.astype(np.float64), 0.0).sum(axis=0)
    means = sums / safe_counts
    sqsums = np.where(mask, host_i8.astype(np.float64) ** 2, 0.0).sum(axis=0)
    var = np.maximum(sqsums / safe_counts - means ** 2, 0.0)
    stds = np.sqrt(var)
    stds = np.where(stds > 0, stds, 1.0)
    means_f32 = means.astype(np.float32)
    stds_f32 = stds.astype(np.float32)

    packed_dev = cp.asarray(packed_host)
    mean_dev = cp.asarray(means_f32)
    std_dev = cp.asarray(stds_f32)
    result = BitpackedDeviceMatrix(
        packed_dev,
        n_samples,
        mean_dev,
        std_dev,
        count_a1=True,
        properties=properties,
    )
    try:
        packed_bytes = int(packed_dev.nbytes)
        side_bytes = int(mean_dev.nbytes) + int(std_dev.nbytes)
        _log(
            f"make_dense_raw_genotype_matrix: bitpacked upgrade: ENGAGED "
            f"(active={n_variants} variants, n_samples={n_samples}, "
            f"packed_bytes={packed_bytes}, side_bytes={side_bytes}, "
            f"HBM bytes={(packed_bytes + side_bytes) / 1e9:.3f} GB)"
        )
    except Exception as _exc:  # noqa: BLE001 - logging never blocks the fast path
        _log(
            f"make_dense_raw_genotype_matrix: bitpacked upgrade: ENGAGED "
            f"(active={n_variants} variants; size log skipped: {_exc!r})"
        )
    return result


# ---------------------------------------------------------------------------
# Hybrid dense + sparse matrix integration point.
#
# The hybrid matrix scaffolding lives in :mod:`sv_pgs.hybrid_matrix` so it can
# be imported without dragging in the bitpacked CUDA kernels. Pipeline code
# that wants to construct a hybrid matrix should import from there directly;
# this re-export keeps the surface discoverable from ``sv_pgs.genotype`` for
# callers that already work with ``RawGenotypeMatrix`` subclasses.
# ---------------------------------------------------------------------------
from sv_pgs.hybrid_matrix import (  # noqa: E402,F401  (re-export integration point)
    BioHybridGenotypeMatrix,
    GpuSparseCarrierMatrix,
    default_carrier_threshold,
)

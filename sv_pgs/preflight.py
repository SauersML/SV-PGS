"""AoU pre-run preflight gate.

This module performs environmental sanity checks BEFORE any GPU initialization,
large staging, or kernel compilation occurs in the sv_pgs pipeline.  It is
deliberately conservative about imports: NO ``cupy`` / ``jax`` import happens
at module load time, only lazily inside :func:`check_aou_preflight`.

If preflight fails (i.e. :attr:`AouPreflightReport.ok` is ``False``), callers
should typically abort with a loud error message via :func:`assert_preflight_ok`.
The function itself returns a structured report instead of raising so callers
can choose to log+continue versus hard-abort.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sv_pgs.gcsfuse_staging import is_gcsfuse_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AouPreflightReport:
    """Structured result of an AoU preflight check.

    Attributes
    ----------
    cdr_storage_path:
        Value of the ``CDR_STORAGE_PATH`` env var (or ``None`` if unset).
    workspace_bucket:
        Value of the ``WORKSPACE_BUCKET`` env var (or ``None`` if unset).
    google_project:
        Value of ``GOOGLE_PROJECT`` or ``GOOGLE_CLOUD_PROJECT`` (first hit).
    cache_dir:
        The cache directory passed to :func:`check_aou_preflight`.
    cache_storage_class:
        Classifier for ``cache_dir``: one of ``"local_hot"``, ``"gcsfuse_mount"``,
        ``"missing"``, or ``"unknown"``.
    free_bytes:
        Free bytes reported by ``shutil.disk_usage`` against ``cache_dir``
        (or its closest existing parent).
    required_stage_bytes:
        The free-space budget required for the staging area.
    required_temp_bytes:
        The free-space budget required for transient temp files.
    cuda_visible_devices:
        Output of ``nvidia-smi -L`` split into one entry per device, or empty
        list if probing failed.
    cupy_available:
        ``True`` iff ``import cupy; cupy.cuda.runtime.getDeviceCount()`` succeeded.
    cupy_devices:
        Number of CuPy-visible devices, or ``0`` on failure.
    jax_preallocate:
        Snapshot of the ``XLA_PYTHON_CLIENT_PREALLOCATE`` env var.
    jax_mem_fraction:
        Snapshot of the ``XLA_PYTHON_CLIENT_MEM_FRACTION`` env var.
    warnings:
        Non-fatal advisories.
    fatal_errors:
        Conditions that should abort the run.
    """

    cdr_storage_path: str | None
    workspace_bucket: str | None
    google_project: str | None
    cache_dir: Path
    cache_storage_class: str  # "local_hot" | "gcsfuse_mount" | "missing" | "unknown"
    free_bytes: int
    required_stage_bytes: int
    required_temp_bytes: int
    cuda_visible_devices: list[str]
    cupy_available: bool
    cupy_devices: int
    jax_preallocate: str | None
    jax_mem_fraction: str | None
    warnings: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when no fatal errors were recorded."""

        return len(self.fatal_errors) == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_cache_dir(cache_dir: Path) -> str:
    """Classify ``cache_dir`` as local-hot, gcsfuse, or unknown."""

    if not cache_dir.exists():
        return "missing"
    try:
        if is_gcsfuse_path(cache_dir):
            return "gcsfuse_mount"
    except Exception:  # noqa: BLE001 - defensive: probe is best effort
        return "unknown"
    return "local_hot"


def _existing_ancestor(path: Path) -> Path:
    """Walk up ``path`` until an existing directory is found."""

    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current


def _probe_nvidia_smi() -> list[str]:
    """Return list of device descriptions from ``nvidia-smi -L`` (or empty)."""

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        completed = subprocess.run(
            [exe, "-L"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    devices: list[str] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            devices.append(line)
    return devices


def _probe_cupy() -> tuple[bool, int, str | None]:
    """Lazy-import cupy and report device count.

    Returns ``(available, device_count, error_message_or_None)``.
    """

    try:
        import cupy  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - any import failure disables CuPy
        return False, 0, f"cupy import failed: {exc!r}"
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001 - runtime probe failure
        return False, 0, f"cupy.cuda.runtime.getDeviceCount() failed: {exc!r}"
    return True, count, None


def _probe_cupy_nvrtc() -> str | None:
    """Compile + launch a no-op CUDA kernel to validate the NVRTC dlopen path.

    ``_probe_cupy`` only calls into the CUDA Runtime (libcudart). It will succeed
    on AoU even when ``libnvrtc.so.12`` is missing — the first real kernel
    compile in production then fails ~40 minutes into the run with
    ``RuntimeError: CuPy failed to load libnvrtc.so.12``. This probe forces an
    NVRTC compile up front so the failure surfaces at preflight time.

    Returns ``None`` on success, or an error string describing the failure.
    """

    try:
        import cupy  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return f"cupy import failed: {exc!r}"
    try:
        kernel = cupy.RawKernel(r'extern "C" __global__ void __sv_pgs_preflight_noop() {}',
                                "__sv_pgs_preflight_noop")
        kernel((1,), (1,), ())
        cupy.cuda.Stream.null.synchronize()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_aou_preflight(
    cache_dir: Path,
    *,
    required_stage_bytes: int = 100_000_000_000,  # 100 GB default safety margin
    required_temp_bytes: int = 50_000_000_000,
    require_gpu: bool = True,
) -> AouPreflightReport:
    """Run all preflight checks for an AoU sv-pgs run.

    This function does NOT import ``cupy`` / ``jax`` at module load — the CuPy
    probe is performed lazily inside this call.  Failures are collected into
    :attr:`AouPreflightReport.fatal_errors` / :attr:`AouPreflightReport.warnings`
    and the report is returned; callers decide whether to abort.

    Parameters
    ----------
    cache_dir:
        Directory that will hold staged genotype caches.  Must be on local
        storage (NOT a gcsfuse mount) for acceptable performance.
    required_stage_bytes:
        Minimum free bytes that must be available on ``cache_dir``'s filesystem.
        Defaults to 100 GB.
    required_temp_bytes:
        Minimum free bytes that must be available for transient temp files.
        Defaults to 50 GB.
    require_gpu:
        When ``True`` (default), missing ``nvidia-smi`` / CuPy is escalated to
        a fatal error.
    """

    cache_dir = Path(cache_dir)
    warnings: list[str] = []
    fatal_errors: list[str] = []

    # 1. cache_dir existence + creation attempt --------------------------------
    if not cache_dir.exists():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fatal_errors.append(
                f"cache_dir {cache_dir!s} does not exist and could not be created: {exc!r}"
            )

    storage_class = _classify_cache_dir(cache_dir)
    if storage_class == "gcsfuse_mount":
        fatal_errors.append(
            f"cache_dir {cache_dir!s} is on a gcsfuse mount; staging there will "
            "destroy throughput. Point cache_dir at a local SSD/NVMe path."
        )
    elif storage_class == "missing":
        # Already covered by the mkdir error above if mkdir failed; otherwise
        # the directory should now exist.  Re-flag defensively.
        if cache_dir.exists():
            storage_class = "local_hot"
        else:
            fatal_errors.append(f"cache_dir {cache_dir!s} does not exist.")

    # 2. Free disk usage -------------------------------------------------------
    probe_target = cache_dir if cache_dir.exists() else _existing_ancestor(cache_dir)
    free_bytes = 0
    try:
        usage = shutil.disk_usage(probe_target)
        free_bytes = int(usage.free)
    except OSError as exc:
        fatal_errors.append(
            f"shutil.disk_usage({probe_target!s}) failed: {exc!r}"
        )

    required_total = int(required_stage_bytes) + int(required_temp_bytes)
    if free_bytes and free_bytes < required_total:
        fatal_errors.append(
            f"Insufficient free space at {probe_target!s}: have {free_bytes} bytes, "
            f"need {required_total} bytes "
            f"(stage={required_stage_bytes} + temp={required_temp_bytes})."
        )

    # 3-5. AoU environment variables ------------------------------------------
    cdr_storage_path = os.environ.get("CDR_STORAGE_PATH")
    if not cdr_storage_path:
        warnings.append("CDR_STORAGE_PATH is not set; AoU CDR resolution may fail.")

    workspace_bucket = os.environ.get("WORKSPACE_BUCKET")
    if not workspace_bucket:
        warnings.append(
            "WORKSPACE_BUCKET is not set; outputs cannot be uploaded to the AoU bucket."
        )

    google_project = (
        os.environ.get("GOOGLE_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    if not google_project:
        warnings.append(
            "Neither GOOGLE_PROJECT nor GOOGLE_CLOUD_PROJECT is set; "
            "requester-pays GCS reads will likely fail."
        )

    # 6. JAX preallocate / mem_fraction ---------------------------------------
    jax_preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    jax_mem_fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")

    preallocate_disabled = (
        jax_preallocate is not None and jax_preallocate.strip().lower() == "false"
    )
    mem_fraction_set = jax_mem_fraction is not None and jax_mem_fraction.strip() != ""
    if not (preallocate_disabled or mem_fraction_set):
        fatal_errors.append(
            "JAX will preallocate 75% of HBM by default; set "
            "XLA_PYTHON_CLIENT_PREALLOCATE=false before sv_pgs imports JAX, "
            "or CuPy will OOM."
        )

    # 7. nvidia-smi probe ------------------------------------------------------
    cuda_devices = _probe_nvidia_smi()
    if require_gpu and not cuda_devices:
        fatal_errors.append(
            "nvidia-smi -L returned no devices (or nvidia-smi missing). "
            "A GPU is required but none is visible."
        )

    # 8. CuPy lazy probe -------------------------------------------------------
    cupy_available, cupy_devices, cupy_err = _probe_cupy()
    if not cupy_available:
        msg = cupy_err or "cupy unavailable"
        if require_gpu:
            fatal_errors.append(
                f"CuPy is required but not usable: {msg}"
            )
        else:
            warnings.append(f"CuPy is not usable (require_gpu=False): {msg}")
    elif require_gpu and cupy_devices == 0:
        fatal_errors.append(
            "CuPy imported but reports 0 visible CUDA devices."
        )
    elif require_gpu and cupy_available:
        # NVRTC dlopen probe — catches the missing libnvrtc.so.12 case which
        # the runtime-only ``_probe_cupy`` cannot detect. See module docs in
        # ``_probe_cupy_nvrtc``.
        nvrtc_err = _probe_cupy_nvrtc()
        if nvrtc_err is not None:
            hint = (
                "  hint: prepend .venv/lib/python3.*/site-packages/nvidia/*/lib "
                "to LD_LIBRARY_PATH before launching sv-pgs (run.sh now does this "
                "automatically; if you launched python directly, source the env "
                "augmentation or invoke via run.sh)."
            )
            fatal_errors.append(
                f"CuPy NVRTC compile probe failed: {nvrtc_err}\n{hint}"
            )

    return AouPreflightReport(
        cdr_storage_path=cdr_storage_path,
        workspace_bucket=workspace_bucket,
        google_project=google_project,
        cache_dir=cache_dir,
        cache_storage_class=storage_class,
        free_bytes=free_bytes,
        required_stage_bytes=int(required_stage_bytes),
        required_temp_bytes=int(required_temp_bytes),
        cuda_visible_devices=cuda_devices,
        cupy_available=cupy_available,
        cupy_devices=cupy_devices,
        jax_preallocate=jax_preallocate,
        jax_mem_fraction=jax_mem_fraction,
        warnings=warnings,
        fatal_errors=fatal_errors,
    )


def assert_preflight_ok(report: AouPreflightReport) -> None:
    """Raise :class:`RuntimeError` listing all fatal errors if not ``report.ok``."""

    if report.ok:
        return
    bullets = "\n".join(f"  - {msg}" for msg in report.fatal_errors)
    raise RuntimeError(
        "AoU preflight failed with the following fatal errors:\n" + bullets
    )


def log_preflight(
    report: AouPreflightReport,
    logger: logging.Logger | None = None,
) -> None:
    """Pretty-print ``report`` to ``logger`` (or stderr if ``logger`` is None)."""

    lines: list[str] = []
    lines.append("=== AoU preflight report ===")
    lines.append(f"  ok                       : {report.ok}")
    lines.append(f"  cache_dir                : {report.cache_dir!s}")
    lines.append(f"  cache_storage_class      : {report.cache_storage_class}")
    lines.append(
        f"  free_bytes               : {report.free_bytes} "
        f"(need stage={report.required_stage_bytes} + temp={report.required_temp_bytes})"
    )
    lines.append(f"  CDR_STORAGE_PATH         : {report.cdr_storage_path!r}")
    lines.append(f"  WORKSPACE_BUCKET         : {report.workspace_bucket!r}")
    lines.append(f"  GOOGLE_PROJECT           : {report.google_project!r}")
    lines.append(
        f"  XLA_PYTHON_CLIENT_PREALLOCATE : {report.jax_preallocate!r}"
    )
    lines.append(
        f"  XLA_PYTHON_CLIENT_MEM_FRACTION: {report.jax_mem_fraction!r}"
    )
    lines.append(f"  nvidia-smi devices       : {report.cuda_visible_devices}")
    lines.append(
        f"  cupy_available           : {report.cupy_available} "
        f"(devices={report.cupy_devices})"
    )
    if report.warnings:
        lines.append("  warnings:")
        for msg in report.warnings:
            lines.append(f"    - {msg}")
    if report.fatal_errors:
        lines.append("  FATAL ERRORS:")
        for msg in report.fatal_errors:
            lines.append(f"    - {msg}")
    text = "\n".join(lines)

    if logger is not None:
        logger.info(text)
    else:
        print(text, file=sys.stderr, flush=True)

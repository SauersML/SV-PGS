"""The one place the fast path learns what hardware it runs on.

Every block size, batch size, prefetch depth and chunk length in the fast
path (dosage store reader, Stage 0 genotype pass, Stage 1 LD-space fit,
Stage 2 exact polish) is derived from a :class:`ComputeBudget`, never from a
hardcoded constant.

The CPU is a first-class device: CUDA is used iff CuPy sees at least one
device. A node that exposes NVIDIA devices but whose CuPy runtime cannot use
them raises instead of silently running on the CPU.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator, Literal

from sv_pgs.progress import log

# Share of the currently free memory a single fast-path job may plan to use.
# The remainder absorbs allocator fragmentation, library workspaces (cuSOLVER,
# cuBLAS, OpenBLAS) and the interpreter itself.
DEVICE_MEMORY_UTILIZATION = 0.85
HOST_MEMORY_UTILIZATION = 0.80


_AUTO_TUNE_HOST_RAM_FALLBACK_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB


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
        3. ``SC_AVPHYS_PAGES * SC_PAGE_SIZE`` (MemFree-equivalent; pessimistic)
        4. 4 GB hardcoded floor
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


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    device_kind: Literal["cpu", "cuda"]
    device_ids: tuple[int, ...]
    device_names: tuple[str, ...]
    device_bytes: tuple[int, ...]
    device_compute_capabilities: tuple[tuple[int, int], ...]
    host_bytes: int
    cpu_threads: int

    @property
    def working_bytes(self) -> int:
        """Bytes one dense per-block job may use on the compute device."""
        if self.device_kind == "cuda":
            return min(self.device_bytes)
        return self.host_bytes

    def describe(self) -> str:
        if self.device_kind == "cpu":
            return (
                f"cpu threads={self.cpu_threads} host_usable={self.host_bytes / 1e9:.1f} GB"
            )
        devices = ", ".join(
            f"{device_id}:{name} sm{major}{minor} {usable / 1e9:.1f} GB"
            for device_id, name, usable, (major, minor) in zip(
                self.device_ids,
                self.device_names,
                self.device_bytes,
                self.device_compute_capabilities,
                strict=True,
            )
        )
        return (
            f"cuda devices=[{devices}] cpu threads={self.cpu_threads} "
            f"host_usable={self.host_bytes / 1e9:.1f} GB"
        )


def _cgroup_memory_headroom_bytes(
    proc_cgroup_file: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    """Bytes the process's memory cgroups still allow, or None when none is limited.

    Slurm jobs and containers cap memory through a cgroup while
    ``/proc/meminfo`` keeps reporting the whole node. The limit can sit on any
    ancestor of the process's own cgroup (Slurm sets it on the job, while the
    step and task cgroups below it are unlimited), so the headroom is the
    smallest ``limit - usage`` over the process's cgroup and every ancestor up
    to the controller root, in cgroup v2 (unified) and v1 (memory controller).
    """
    if not proc_cgroup_file.exists():
        return None
    headroom: int | None = None
    for line in proc_cgroup_file.read_text(encoding="utf-8").splitlines():
        hierarchy_id, controllers, relative_path = line.split(":", 2)
        if hierarchy_id == "0" and controllers == "":
            controller_root = cgroup_root
            limit_name, usage_name = "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            controller_root = cgroup_root / "memory"
            limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        group = controller_root / relative_path.strip().lstrip("/")
        for level in (group, *group.parents):
            limit_file, usage_file = level / limit_name, level / usage_name
            if limit_file.exists() and usage_file.exists():
                limit_text = limit_file.read_text(encoding="utf-8").strip()
                # cgroup v2 writes "max" and v1 a huge page-aligned sentinel for "unlimited".
                if limit_text != "max" and int(limit_text) < 1 << 60:
                    usage_bytes = int(usage_file.read_text(encoding="utf-8").strip())
                    level_headroom = max(int(limit_text) - usage_bytes, 0)
                    headroom = level_headroom if headroom is None else min(headroom, level_headroom)
            if level == controller_root:
                break
    return headroom


def _usable_host_bytes() -> int:
    available_bytes = int(_detect_available_host_ram_bytes())
    cgroup_headroom = _cgroup_memory_headroom_bytes()
    if cgroup_headroom is not None:
        available_bytes = min(available_bytes, cgroup_headroom)
    return int(available_bytes * HOST_MEMORY_UTILIZATION)


def _nvidia_devices_exposed() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return visible.strip() not in ("", "-1")
    return Path("/dev/nvidiactl").exists()


def detect_compute_budget() -> ComputeBudget:
    cpu_threads = len(os.sched_getaffinity(0))
    host_bytes = _usable_host_bytes()
    cupy = _try_import_cupy()
    if cupy is None:
        if _nvidia_devices_exposed():
            raise RuntimeError(
                "NVIDIA devices are exposed to this process but CuPy cannot use them: "
                + _cupy_runtime_diagnostic()
                + " | "
                + _nvidia_driver_diagnostic()
            )
        budget = ComputeBudget(
            device_kind="cpu",
            device_ids=(),
            device_names=(),
            device_bytes=(),
            device_compute_capabilities=(),
            host_bytes=host_bytes,
            cpu_threads=cpu_threads,
        )
        log("  compute budget: " + budget.describe())
        return budget
    device_ids = tuple(range(int(cupy.cuda.runtime.getDeviceCount())))
    names: list[str] = []
    usable: list[int] = []
    capabilities: list[tuple[int, int]] = []
    for device_id in device_ids:
        with _cupy_device_context(cupy, device_id):
            cupy.get_default_memory_pool().free_all_blocks()
            free_bytes, _total_bytes = cupy.cuda.runtime.memGetInfo()
            properties = cupy.cuda.runtime.getDeviceProperties(device_id)
        name = properties["name"]
        names.append(name.decode("utf-8") if isinstance(name, bytes) else str(name))
        usable.append(int(int(free_bytes) * DEVICE_MEMORY_UTILIZATION))
        capabilities.append((int(properties["major"]), int(properties["minor"])))
    budget = ComputeBudget(
        device_kind="cuda",
        device_ids=device_ids,
        device_names=tuple(names),
        device_bytes=tuple(usable),
        device_compute_capabilities=tuple(capabilities),
        host_bytes=host_bytes,
        cpu_threads=cpu_threads,
    )
    log("  compute budget: " + budget.describe())
    return budget

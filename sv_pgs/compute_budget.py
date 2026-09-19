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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sv_pgs.genotype import (
    _cupy_device_context,
    _cupy_runtime_diagnostic,
    _detect_available_host_ram_bytes,
    _nvidia_driver_diagnostic,
    _try_import_cupy,
)
from sv_pgs.progress import log

# Share of the currently free memory a single fast-path job may plan to use.
# The remainder absorbs allocator fragmentation, library workspaces (cuSOLVER,
# cuBLAS, OpenBLAS) and the interpreter itself.
DEVICE_MEMORY_UTILIZATION = 0.85
HOST_MEMORY_UTILIZATION = 0.80


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

"""Lightweight progress logging for pipeline diagnostics."""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import time

_start_time: float = time.monotonic()


def mem() -> str:
    """Return current RSS in MB (Linux: from /proc, macOS: peak RSS)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return line.split()[1] + " kB->~" + str(int(line.split()[1]) // 1024) + " MB"
    except (OSError, IndexError):
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss //= 1024 * 1024
    else:
        rss //= 1024
    return f"{rss} MB"


def elapsed() -> str:
    """Return wall-clock time since module import."""
    dt = time.monotonic() - _start_time
    if dt < 60:
        return f"{dt:.1f}s"
    minutes = int(dt // 60)
    seconds = dt % 60
    if minutes < 60:
        return f"{minutes}m{seconds:.0f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes}m{seconds:.0f}s"


def log(message: str) -> None:
    """Print a timestamped progress message to stderr."""
    print(f"[{elapsed()} | {mem()}] {message}", file=sys.stderr, flush=True)


def _format_bytes(value: object) -> str:
    try:
        size = float(str(value))
    except (TypeError, ValueError):
        return str(value)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.1f}{units[unit_index]}"


def gpu_memory_snapshot() -> str:
    try:
        import jax
    except Exception as error:
        return f"jax_unavailable={error}"
    try:
        devices = jax.devices()
    except Exception as error:
        return f"device_query_failed={error}"
    if not devices:
        return "devices=[]"
    parts: list[str] = []
    for device in devices:
        part = f"{device.platform}:{getattr(device, 'id', '?')}:{getattr(device, 'device_kind', 'unknown')}"
        memory_stats_getter = getattr(device, "memory_stats", None)
        if callable(memory_stats_getter):
            try:
                stats = memory_stats_getter() or {}
                stat_parts = []
                for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                    if key in stats:
                        stat_parts.append(f"{key}={_format_bytes(stats[key])}")
                if stat_parts:
                    part += "[" + ",".join(stat_parts) + "]"
            except Exception as error:
                part += f"[memory_stats_error={error}]"
        parts.append(part)
    return "devices=" + "; ".join(parts)


def nvidia_smi_snapshot() -> str:
    command = shutil.which("nvidia-smi")
    if command is None:
        return "nvidia-smi=unavailable"
    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception as error:
        return f"nvidia-smi_error={error}"
    if result.returncode != 0:
        stderr = result.stderr.strip().replace("\n", " | ")
        return f"nvidia-smi_rc={result.returncode} stderr={stderr}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return "nvidia-smi=no_visible_gpus"
    return " | ".join(lines)


def jax_runtime_snapshot() -> str:
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
    )
    env_summary = ", ".join(f"{key}={os.environ.get(key, '<unset>')}" for key in env_keys)
    try:
        import jax
        import jaxlib
        backend = jax.default_backend()
        devices = jax.devices()
        version_summary = f"jax={jax.__version__} jaxlib={jaxlib.__version__} backend={backend}"
        device_summary = ", ".join(
            f"{device.platform}:{getattr(device, 'id', '?')}:{getattr(device, 'device_kind', 'unknown')}"
            for device in devices
        )
        if not device_summary:
            device_summary = "<none>"
        return f"{version_summary} devices=[{device_summary}] env=[{env_summary}]"
    except Exception as error:
        return f"jax_runtime_error={error} env=[{env_summary}]"

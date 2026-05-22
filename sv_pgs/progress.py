"""Lightweight progress logging for pipeline diagnostics."""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback

_start_time: float = time.monotonic()
_log_file = None

_heartbeat_thread: threading.Thread | None = None
_heartbeat_stop: threading.Event | None = None
# Track CPU usage between heartbeats so we can report a delta-utilization
# instead of a meaningless cumulative number.
_heartbeat_last_wall: float | None = None
_heartbeat_last_cpu: float | None = None
_heartbeat_last_nvidia_smi: float = 0.0
_heartbeat_last_nvidia_smi_value: str = "nvidia-smi=not-yet-sampled"


def set_log_file(path: str | os.PathLike) -> None:
    """Set a file path where all log() output is mirrored."""
    global _log_file
    if _log_file is not None:
        try:
            _log_file.close()
        except OSError:
            pass
    _log_file = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered


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
    """Print a timestamped progress message to stderr and log file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} | {elapsed()} | {mem()}] {message}"
    print(line, file=sys.stderr, flush=True)
    if _log_file is not None:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except OSError:
            pass


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
    except (ImportError, RuntimeError) as error:
        return f"jax_unavailable={error}"
    try:
        devices = jax.devices()
    except RuntimeError as error:
        return f"device_query_failed={error}"
    if not devices:
        return "devices=[]"
    parts: list[str] = []
    for device in devices:
        part = f"{device.platform}:{getattr(device, 'id', '?')}:{getattr(device, 'device_kind', 'unknown')}"
        memory_stats_getter = getattr(device, "memory_stats", None)
        if callable(memory_stats_getter):
            try:
                raw_stats = memory_stats_getter() or {}
                stats: dict[str, object] = dict(raw_stats) if isinstance(raw_stats, dict) else {}
                stat_parts = []
                for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                    if key in stats:
                        stat_parts.append(f"{key}={_format_bytes(stats[key])}")
                if stat_parts:
                    part += "[" + ",".join(stat_parts) + "]"
            except (RuntimeError, KeyError) as error:
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
    except (OSError, subprocess.SubprocessError) as error:
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
        jaxlib_version = getattr(jaxlib, "__version__", "<unknown>")
        version_summary = f"jax={jax.__version__} jaxlib={jaxlib_version} backend={backend}"
        device_summary = ", ".join(
            f"{device.platform}:{getattr(device, 'id', '?')}:{getattr(device, 'device_kind', 'unknown')}"
            for device in devices
        )
        if not device_summary:
            device_summary = "<none>"
        return f"{version_summary} devices=[{device_summary}] env=[{env_summary}]"
    except (ImportError, RuntimeError) as error:
        return f"jax_runtime_error={error} env=[{env_summary}]"


# -- Heartbeat / stack sampler --------------------------------------------
#
# Background thread that emits a periodic "where am I now" snapshot so a stall
# in the main thread is no longer invisible. Each heartbeat reports:
#   * Top-of-stack frames for the main thread (file:line:func)
#   * Process CPU% over the heartbeat interval
#   * Resident memory
#   * Active OS thread count and Python thread count
#   * GPU: nvidia-smi util/mem (sampled at most every 30 s; it shells out)
#   * Cupy default mempool used/total bytes (cheap; in-process)
#
# Designed to be lightweight and crash-resistant: any exception inside the
# heartbeat is caught and logged, so a transient failure (e.g. cupy not
# initialized yet) cannot kill the run.


def _format_thread_frame(frame) -> list[str]:
    """Return the deepest ~6 frames of `frame` as 'file:line:func' strings."""
    if frame is None:
        return ["<no frame>"]
    frames: list[str] = []
    cursor = frame
    while cursor is not None and len(frames) < 6:
        code = cursor.f_code
        filename = os.path.basename(code.co_filename) or code.co_filename
        frames.append(f"{filename}:{cursor.f_lineno}:{code.co_name}")
        cursor = cursor.f_back
    return frames


def _process_cpu_seconds() -> float:
    """Return cumulative user+system CPU seconds for this process."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return float(rusage.ru_utime) + float(rusage.ru_stime)


def _cupy_mempool_snapshot() -> str:
    try:
        import cupy
    except ImportError:
        return "cupy=unavailable"
    try:
        pool = cupy.get_default_memory_pool()
        used = int(pool.used_bytes())
        total = int(pool.total_bytes())
        pinned = cupy.get_default_pinned_memory_pool()
        pinned_used = int(pinned.n_free_blocks())
        return (
            f"cupy_used={_format_bytes(used)} "
            + f"cupy_pool={_format_bytes(total)} "
            + f"cupy_pinned_free_blocks={pinned_used}"
        )
    except (RuntimeError, AttributeError) as error:
        return f"cupy_mempool_error={error}"


def _maybe_refresh_nvidia_smi(now: float) -> str:
    """Sample nvidia-smi at most every 30 seconds (shelling out is expensive)."""
    global _heartbeat_last_nvidia_smi, _heartbeat_last_nvidia_smi_value
    if now - _heartbeat_last_nvidia_smi > 30.0:
        _heartbeat_last_nvidia_smi_value = nvidia_smi_snapshot()
        _heartbeat_last_nvidia_smi = now
    return _heartbeat_last_nvidia_smi_value


def _loadavg_snapshot() -> str:
    try:
        one, five, fifteen = os.getloadavg()
        return f"load1={one:.2f} load5={five:.2f} load15={fifteen:.2f}"
    except (OSError, AttributeError):
        return "load=unknown"


def _heartbeat_tick(interval_seconds: float, main_thread_id: int) -> None:
    """Emit one heartbeat. Designed to never raise."""
    global _heartbeat_last_wall, _heartbeat_last_cpu
    try:
        now = time.monotonic()
        # CPU utilization since last tick.
        cpu_now = _process_cpu_seconds()
        if _heartbeat_last_wall is None or _heartbeat_last_cpu is None:
            cpu_pct_str = "cpu=warming"
        else:
            wall_delta = max(now - _heartbeat_last_wall, 1e-6)
            cpu_delta = cpu_now - _heartbeat_last_cpu
            cpu_pct = 100.0 * cpu_delta / wall_delta
            cpu_pct_str = f"cpu={cpu_pct:.0f}% ({cpu_delta:.1f}s in {wall_delta:.1f}s)"
        _heartbeat_last_wall = now
        _heartbeat_last_cpu = cpu_now

        frames = sys._current_frames()
        main_frame = frames.get(main_thread_id)
        stack_lines = _format_thread_frame(main_frame)
        # Other threads that are not the main one and not the heartbeat itself.
        self_ident = threading.get_ident()
        background_threads = [
            tid for tid in frames if tid != main_thread_id and tid != self_ident
        ]

        nvidia = _maybe_refresh_nvidia_smi(now)
        cupy_info = _cupy_mempool_snapshot()
        loadavg = _loadavg_snapshot()
        thread_count = threading.active_count()

        # Print in a single contiguous block so it's easy to grep.
        log(
            "HEARTBEAT @ "
            + f"interval={interval_seconds:.0f}s  {cpu_pct_str}  mem={mem()}  "
            + f"threads={thread_count} (bg={len(background_threads)})  {loadavg}"
        )
        log("  main-thread stack (deepest first):")
        for frame_line in stack_lines:
            log(f"    {frame_line}")
        log(f"  gpu: {nvidia}")
        log(f"  gpu_mempool: {cupy_info}")
    except (OSError, RuntimeError, ValueError) as error:
        # Never crash the heartbeat thread.
        try:
            log(f"HEARTBEAT error: {error}")
        except OSError:
            pass


def _heartbeat_loop(interval_seconds: float, main_thread_id: int) -> None:
    assert _heartbeat_stop is not None
    while not _heartbeat_stop.is_set():
        _heartbeat_tick(interval_seconds, main_thread_id)
        # Wait up to interval_seconds, but break early if stop is set.
        _heartbeat_stop.wait(interval_seconds)


def start_heartbeat(interval_seconds: float = 60.0) -> None:
    """Start a background thread that emits periodic stack/CPU/GPU snapshots.

    Safe to call more than once; subsequent calls are no-ops.
    """
    global _heartbeat_thread, _heartbeat_stop
    global _heartbeat_last_wall, _heartbeat_last_cpu
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        return
    _heartbeat_stop = threading.Event()
    _heartbeat_last_wall = None
    _heartbeat_last_cpu = None
    main_thread_id = threading.main_thread().ident
    assert main_thread_id is not None
    thread = threading.Thread(
        target=_heartbeat_loop,
        args=(float(interval_seconds), int(main_thread_id)),
        name="sv-pgs-heartbeat",
        daemon=True,
    )
    _heartbeat_thread = thread
    thread.start()
    log(f"heartbeat sampler started (every {interval_seconds:.0f}s)")


def stop_heartbeat() -> None:
    """Stop the background heartbeat thread (best-effort)."""
    global _heartbeat_thread, _heartbeat_stop
    if _heartbeat_stop is not None:
        _heartbeat_stop.set()
    if _heartbeat_thread is not None:
        _heartbeat_thread.join(timeout=2.0)
    _heartbeat_thread = None
    _heartbeat_stop = None


# Keep `traceback` import used in case future variants want a full formatted
# traceback string. Reference it so static analyzers don't flag it as unused.
_ = traceback

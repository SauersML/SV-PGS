import importlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

# Do not let JAX reserve nearly all VRAM up front.  The training pipeline keeps
# large genotype matrices resident on device, and the compiler also needs its
# own scratch space during the first dense-kernel builds.
# Do not let JAX/XLA pre-allocate a large GPU memory pool.  CuPy manages
# GPU memory for genotype matrices and linear algebra.  JAX only needs
# scratch space for element-wise ops and XLA compilation.
# Assign unconditionally — using setdefault would let an inherited unsafe value
# (e.g. PREALLOCATE=true or MEM_FRACTION=0.9) silently win and starve CuPy.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"


def _most_free_cuda_device() -> tuple[int, int, int] | None:
    """Return the currently most-free CUDA device without hiding other GPUs."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return None
    try:
        import cupy as _cupy  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError):
        return None
    try:
        device_count = int(_cupy.cuda.runtime.getDeviceCount())
    except (AttributeError, OSError, RuntimeError):
        return None
    if device_count < 1:
        return None
    best_id = 0
    best_free = -1
    best_total = 0
    for device_id in range(device_count):
        try:
            with _cupy.cuda.Device(device_id):
                free, total = _cupy.cuda.runtime.memGetInfo()
        except (AttributeError, OSError, RuntimeError):
            continue
        if int(free) > best_free:
            best_free = int(free)
            best_total = int(total)
            best_id = device_id
    if best_free < 0:
        return None
    return best_id, best_free, best_total


def _self_pin_cuda_visible_device() -> tuple[int, int, int] | None:
    """Pin this process to the most-free CUDA device by setting
    ``CUDA_VISIBLE_DEVICES`` BEFORE jax (and cupy) initialize.

    No-op when ``CUDA_VISIBLE_DEVICES`` is already set (caller — typically
    ``aou_runner`` subprocess orchestration — has already pinned), or when
    fewer than 2 devices are visible (single-GPU box doesn't need it).

    Returns the ``(device_id, free_bytes, total_bytes)`` triple of the chosen
    device so callers (banner / ``require_gpu``) can report it; returns
    ``None`` when no self-pin occurred or no GPU is usable.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return _most_free_cuda_device()
    try:
        import cupy as _cupy  # type: ignore[import-not-found]
    except (ImportError, OSError, RuntimeError):
        return None
    try:
        device_count = int(_cupy.cuda.runtime.getDeviceCount())
    except (AttributeError, OSError, RuntimeError):
        return None
    if device_count < 2:
        # Single-GPU (or no-GPU) box: pinning gives nothing and would only
        # add a surprising env mutation. Just probe and return.
        return _most_free_cuda_device()
    probed = _most_free_cuda_device()
    if probed is None:
        return None
    device_id, free_bytes, total_bytes = probed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Race-safety: re-probe to confirm cupy now sees exactly one device.
    # CuPy/CUDA driver init may have already cached the device list before
    # this env mutation; if so, log a warning so operators can restart with
    # the env pre-set if it actually misbehaves on this host.
    try:
        post_count = int(_cupy.cuda.runtime.getDeviceCount())
    except (AttributeError, OSError, RuntimeError):
        post_count = -1
    print(
        f"GPU pin: selected device {device_id} of {device_count} "
        f"(free={free_bytes / 1e9:.1f}/{total_bytes / 1e9:.1f} GB) "
        f"→ CUDA_VISIBLE_DEVICES={device_id}",
        flush=True,
    )
    if post_count not in (-1, 1):
        print(
            f"GPU pin: WARNING post-pin cupy reports {post_count} devices "
            f"(expected 1); CUDA may have initialized before pin took effect. "
            f"Re-run with CUDA_VISIBLE_DEVICES={device_id} set in the parent "
            f"environment to force pinning before any CUDA init.",
            flush=True,
        )
    return device_id, free_bytes, total_bytes


SELECTED_CUDA_DEVICE: tuple[int, int, int] | None = _self_pin_cuda_visible_device()

# Work around XLA compilation segfaults on Turing GPUs (T4).  These flags make
# JAX materially slower on newer GPUs, so only enable them when a Turing-class
# device is actually present.  See:
#   https://github.com/jax-ml/jax/issues/17349  (Turing GPU segfaults)
#   https://github.com/jax-ml/jax/issues/29139  (backend_compile segfault)
_TURING_WORKAROUND_FLAGS = (
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_force_compilation_parallelism=1",
)


def _split_nvidia_smi_lines(output: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _is_turing_gpu(
    device_names: Sequence[str],
    compute_capabilities: Sequence[str],
) -> bool:
    if any(str(capability).strip().startswith("7.5") for capability in compute_capabilities):
        return True
    return any("t4" in str(device_name).lower() or "turing" in str(device_name).lower() for device_name in device_names)


def _parse_compute_capability_major(capability: str) -> int:
    try:
        return int(str(capability).strip().split(".", 1)[0])
    except (ValueError, IndexError):
        return 0


def _is_ampere_or_newer(compute_capabilities: Sequence[str]) -> bool:
    return any(_parse_compute_capability_major(capability) >= 8 for capability in compute_capabilities)


def _is_hopper_or_newer(compute_capabilities: Sequence[str]) -> bool:
    return any(_parse_compute_capability_major(capability) >= 9 for capability in compute_capabilities)


def _query_nvidia_smi(field: str) -> tuple[str, ...]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return ()
    try:
        result = subprocess.run(
            [command, f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    return _split_nvidia_smi_lines(result.stdout)


@lru_cache(maxsize=1)
def _compute_capabilities() -> tuple[str, ...]:
    return _query_nvidia_smi("compute_cap")


@lru_cache(maxsize=1)
def _device_names() -> tuple[str, ...]:
    return _query_nvidia_smi("name")


@lru_cache(maxsize=1)
def turing_workarounds_enabled() -> bool:
    return _is_turing_gpu(
        device_names=_device_names(),
        compute_capabilities=_compute_capabilities(),
    )


@lru_cache(maxsize=1)
def tensor_core_matmul_enabled() -> bool:
    """True on Ampere (SM 8.0) and newer, where TF32 tensor cores accelerate
    float32 matmul with float32 accumulation. Mantissa truncation is below the
    float32 noise floor for standardized genotype workloads, so this is
    quality-preserving."""
    return _is_ampere_or_newer(_compute_capabilities())


@lru_cache(maxsize=1)
def hopper_or_newer_gpu_present() -> bool:
    """True when at least one Hopper-class (SM 9.0+) GPU is visible.

    Wired as a public hook so downstream tuning code (preconditioner ranks,
    Gram-tile work targets, batch sizes) can opt into H100-friendly defaults
    without re-querying nvidia-smi. Currently used to lift the CuPy memory
    pool's growth-release log threshold on devices large enough that the T4
    sized constant is noise, and reserved for future SM90 fp16-mma kernel
    enablement.
    """
    return _is_hopper_or_newer(_compute_capabilities())


if turing_workarounds_enabled():
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    for flag in _TURING_WORKAROUND_FLAGS:
        if flag not in _xla_flags:
            _xla_flags = f"{_xla_flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = _xla_flags

# Enable CuPy's TF32 fast path on Ampere+. Honoured the next time CuPy is
# imported anywhere in the process. cuBLAS computes matmul in TF32 (10-bit
# mantissa) with float32 accumulation; the truncation sits below the float32
# noise floor for standardized genotype-style inputs, and Gram products /
# Cholesky factorizations stay in float64 regardless.
if tensor_core_matmul_enabled():
    os.environ.setdefault("CUPY_TF32", "1")

# JAX must be imported AFTER the XLA env-var setup above (otherwise the JAX
# runtime captures the wrong settings). Use importlib so static-analysis tools
# don't flag these as misplaced top-of-file imports.
jax_config = importlib.import_module("jax").config

# Enable 64-bit precision (required for Bayesian inference numerics).
jax_config.update("jax_enable_x64", True)

# Prefer full float32 matmul precision in JAX. T4/Turing has no TF32 hardware,
# so this is speed-neutral there; on newer GPUs it avoids silently using TF32 in
# JAX code paths that affect optimizer state.
jax_config.update("jax_default_matmul_precision", "highest")

# jax.numpy must be imported AFTER jax_config.update above.
jnp = importlib.import_module("jax.numpy")


def _cupy_runtime_status() -> tuple[bool, str]:
    try:
        import cupy as cp
    except (ImportError, OSError, RuntimeError) as exc:
        return False, f"cupy_unavailable={type(exc).__name__}: {exc}"
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except (AttributeError, OSError, RuntimeError) as exc:
        return False, f"cupy_cuda_error={type(exc).__name__}: {exc}"
    if device_count < 1:
        return False, "cupy_devices=0"
    return True, f"cupy_devices={device_count}"


def gpu_float32_compute_enabled() -> bool:
    """Use float32 for GPU matmul on any GPU with CuPy available.

    Float32 is correct for genotype matmul on all GPUs — the Gram matrix
    accumulates results in float64, and Cholesky always runs in float64.
    Float64 matmul is catastrophically slow on consumer/datacenter GPUs
    (P4 ~0.1 TFLOPS, T4 ~0.25 TFLOPS, V100 ~7 TFLOPS vs 14 TFLOPS float32)
    with no accuracy benefit for this workload.
    """
    cupy_ok, _ = _cupy_runtime_status()
    return cupy_ok


def jax_dense_linear_algebra_preferred() -> bool:
    """Whether dense linear algebra should prefer JAX over backend-specific kernels."""
    return not turing_workarounds_enabled()


def gpu_compute_numpy_dtype() -> np.dtype[np.floating[Any]]:
    return np.dtype(np.float32 if gpu_float32_compute_enabled() else np.float64)


def gpu_compute_jax_dtype() -> Any:
    return jnp.float32 if gpu_float32_compute_enabled() else jnp.float64

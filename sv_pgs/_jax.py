import os
import shutil
import subprocess
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

# Do not let JAX reserve nearly all VRAM up front.  The training pipeline keeps
# large genotype matrices resident on device, and the compiler also needs its
# own scratch space during the first dense-kernel builds.
# Do not let JAX/XLA pre-allocate a large GPU memory pool.  CuPy manages
# GPU memory for genotype matrices and linear algebra.  JAX only needs
# scratch space for element-wise ops and XLA compilation.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

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
def turing_workarounds_enabled() -> bool:
    return _is_turing_gpu(
        device_names=_query_nvidia_smi("name"),
        compute_capabilities=_query_nvidia_smi("compute_cap"),
    )


if turing_workarounds_enabled():
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    for flag in _TURING_WORKAROUND_FLAGS:
        if flag not in _xla_flags:
            _xla_flags = f"{_xla_flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = _xla_flags

from jax import config as jax_config  # must follow XLA env-var setup above

# Enable 64-bit precision (required for Bayesian inference numerics).
jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # must follow jax_config.update above


def _cupy_runtime_status() -> tuple[bool, str]:
    try:
        import cupy as cp
    except (ImportError, OSError, RuntimeError) as exc:
        return False, f"cupy_unavailable={type(exc).__name__}: {exc}"
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except (OSError, RuntimeError) as exc:
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


def gpu_compute_numpy_dtype() -> np.dtype:
    return np.dtype(np.float32 if gpu_float32_compute_enabled() else np.float64)


def gpu_compute_jax_dtype():
    return jnp.float32 if gpu_float32_compute_enabled() else jnp.float64

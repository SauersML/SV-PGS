import os

# Do not let JAX reserve nearly all VRAM up front.  The training pipeline keeps
# large genotype matrices resident on device, and the compiler also needs its
# own scratch space during the first dense-kernel builds.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

# Disable Triton GEMM and autotuning to work around XLA compilation segfaults
# on some GPU environments (e.g. T4 with certain CUDA driver versions).
_xla_flags = os.environ.get("XLA_FLAGS", "")
for flag in [
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_autotune_level=0",
]:
    if flag not in _xla_flags:
        _xla_flags = f"{_xla_flags} {flag}".strip()
os.environ["XLA_FLAGS"] = _xla_flags

from jax import config as jax_config

# Enable 64-bit precision (required for Bayesian inference numerics).
jax_config.update("jax_enable_x64", True)

try:
    import jax

    _default_device = jax.default_backend()
except Exception:
    _default_device = "cpu"

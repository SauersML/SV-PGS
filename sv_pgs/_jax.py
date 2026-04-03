import os

# Do not let JAX reserve nearly all VRAM up front.  The training pipeline keeps
# large genotype matrices resident on device, and the compiler also needs its
# own scratch space during the first dense-kernel builds.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

# Work around XLA compilation segfaults on Turing GPUs (T4) and other
# environments.  See:
#   https://github.com/jax-ml/jax/issues/17349  (Turing GPU segfaults)
#   https://github.com/jax-ml/jax/issues/29139  (backend_compile segfault)
_xla_flags = os.environ.get("XLA_FLAGS", "")
for flag in [
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_autotune_level=0",
    "--xla_gpu_force_compilation_parallelism=1",
]:
    if flag not in _xla_flags:
        _xla_flags = f"{_xla_flags} {flag}".strip()
os.environ["XLA_FLAGS"] = _xla_flags

from jax import config as jax_config

# Enable 64-bit precision (required for Bayesian inference numerics).
jax_config.update("jax_enable_x64", True)

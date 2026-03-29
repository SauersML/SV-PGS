import os
import warnings

from jax import config as jax_config

# Enable 64-bit precision (required for Bayesian inference numerics).
jax_config.update("jax_enable_x64", True)

# Auto-detect GPU: if available, JAX will use it automatically.
# Log which device JAX selected.
try:
    import jax
    _default_device = jax.default_backend()
    if _default_device == "gpu":
        # Pre-allocate less GPU memory to leave room for bed_reader and numpy.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
except Exception:
    _default_device = "cpu"

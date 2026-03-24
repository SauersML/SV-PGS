"""Shared numerical utilities using JAX."""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def stable_sigmoid(values: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable sigmoid using JAX's built-in implementation."""
    return jax.nn.sigmoid(values)

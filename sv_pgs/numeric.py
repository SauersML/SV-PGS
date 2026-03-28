"""Shared numerical utilities."""
from __future__ import annotations

import sv_pgs._jax  # noqa: F401

import jax.numpy as jnp


def stable_sigmoid(values) -> jnp.ndarray:
    """Numerically stable sigmoid on JAX arrays."""
    value_array = jnp.asarray(values, dtype=jnp.float64)
    positive_branch = value_array >= 0.0
    negative_exponential = jnp.exp(jnp.where(positive_branch, -value_array, value_array))
    return jnp.where(
        positive_branch,
        1.0 / (1.0 + negative_exponential),
        negative_exponential / (1.0 + negative_exponential),
    )

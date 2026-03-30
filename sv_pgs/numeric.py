"""Shared numerical utilities."""
from __future__ import annotations

import sv_pgs._jax  # noqa: F401

import jax.numpy as jnp


def stable_sigmoid(values) -> jnp.ndarray:
    """Convert a real-valued score to a probability between 0 and 1.

    sigmoid(x) = 1 / (1 + exp(-x))

    Used in binary trait models: a positive score means higher disease risk,
    and the sigmoid maps that to a probability (e.g. 0.73 = 73% risk).

    The "stable" part: naive computation of exp(-x) overflows for large
    negative x.  We split into two branches:
      - x >= 0: compute exp(-x) directly (safe, since -x <= 0)
      - x <  0: compute exp(x) and rearrange to avoid overflow
    Both branches give the same mathematical result, just different
    floating-point paths for numerical safety.
    """
    value_array = jnp.asarray(values, dtype=jnp.float64)
    positive_branch = value_array >= 0.0
    negative_exponential = jnp.exp(jnp.where(positive_branch, -value_array, value_array))
    return jnp.where(
        positive_branch,
        1.0 / (1.0 + negative_exponential),
        negative_exponential / (1.0 + negative_exponential),
    )

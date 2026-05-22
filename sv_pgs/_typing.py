"""Shared type aliases for numpy and JAX arrays.

Centralizes the parameterized array types we use across the package. With
mypy --strict, every bare ``np.ndarray`` is flagged as missing generic
parameters; importing these aliases gives a single principled source of
truth instead of sprinkling ``ndarray[Any, np.dtype[...]]`` everywhere.

Conventions:
- ``F32Array``, ``F64Array``, ``I8Array``, ``I32Array``, ``I64Array``,
  ``BoolArray`` are 1-D-or-N-D ``np.ndarray`` with the named dtype.
- ``NDArray`` (= ``np.typing.NDArray[np.generic]``) is the broadest
  "some numpy array" alias; reach for it only when a function genuinely
  accepts any dtype.
- ``JaxArray`` covers both ``jax.Array`` and the transitional
  ``jaxlib.xla_extension.ArrayImpl`` so we can annotate JAX-traced values
  without coupling to the JAX import at module top-level.

Keep this module dependency-light: no JAX-time eager imports, just
``np.typing`` and ``typing``.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

# Generic numpy arrays --------------------------------------------------------

NDArray: TypeAlias = npt.NDArray[Any]
"""Any numpy array (dtype unconstrained)."""

F32Array: TypeAlias = npt.NDArray[np.float32]
F64Array: TypeAlias = npt.NDArray[np.float64]
I8Array: TypeAlias = npt.NDArray[np.int8]
I16Array: TypeAlias = npt.NDArray[np.int16]
I32Array: TypeAlias = npt.NDArray[np.int32]
I64Array: TypeAlias = npt.NDArray[np.int64]
U8Array: TypeAlias = npt.NDArray[np.uint8]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# JAX arrays ------------------------------------------------------------------
# We don't import jax at module load to keep cold-start cheap and to let
# tests run without JAX installed. ``JaxArray`` therefore aliases to ``Any``
# at runtime; the typing alias still narrows callers in mypy's view.

JaxArray: TypeAlias = Any
"""A JAX-traced array (``jax.Array`` or ``ArrayImpl``)."""

__all__ = [
    "BoolArray",
    "F32Array",
    "F64Array",
    "I8Array",
    "I16Array",
    "I32Array",
    "I64Array",
    "JaxArray",
    "NDArray",
    "U8Array",
]

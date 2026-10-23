"""NumPy array aliases by dtype; dimensions are specified by each caller.

``NDArray`` accepts any dtype. The other aliases name a specific NumPy dtype.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

NDArray: TypeAlias = npt.NDArray[Any]
"""Any numpy array (dtype unconstrained)."""

F32Array: TypeAlias = npt.NDArray[np.float32]
F64Array: TypeAlias = npt.NDArray[np.float64]
I32Array: TypeAlias = npt.NDArray[np.int32]
I64Array: TypeAlias = npt.NDArray[np.int64]
U8Array: TypeAlias = npt.NDArray[np.uint8]
U16Array: TypeAlias = npt.NDArray[np.uint16]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

__all__ = [
    "BoolArray",
    "F32Array",
    "F64Array",
    "I32Array",
    "I64Array",
    "NDArray",
    "U16Array",
    "U8Array",
]

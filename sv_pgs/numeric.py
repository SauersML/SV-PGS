"""Shared numerical utilities."""
from __future__ import annotations

import numpy as np


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid on NumPy arrays."""
    value_array = np.asarray(values, dtype=np.float64)
    positive_branch = value_array >= 0.0
    negative_exponential = np.exp(np.where(positive_branch, -value_array, value_array))
    sigmoid_values = np.where(
        positive_branch,
        1.0 / (1.0 + negative_exponential),
        negative_exponential / (1.0 + negative_exponential),
    )
    return sigmoid_values.astype(np.float32, copy=False)

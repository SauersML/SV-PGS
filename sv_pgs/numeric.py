from __future__ import annotations

import numpy as np


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.asarray(np.clip(values, -80.0, 80.0), dtype=np.float64)
    positive_mask = clipped_values >= 0.0
    negative_mask = ~positive_mask
    output = np.empty_like(clipped_values, dtype=np.float64)
    output[positive_mask] = 1.0 / (1.0 + np.exp(-clipped_values[positive_mask]))
    exp_values = np.exp(clipped_values[negative_mask])
    output[negative_mask] = exp_values / (1.0 + exp_values)
    return output.astype(np.float64)

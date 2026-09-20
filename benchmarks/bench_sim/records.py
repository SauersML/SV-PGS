"""The measured record set and the streaming row block, with no dependency beyond numpy.

The harness and the submission-facing modules import from here, so a submission's environment needs only
numpy and scipy (the measurement arms themselves also need cyvcf2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

STREAM_ROWS = 20_000
# aou2's reference panel keeps records with allele count >= 2 (imputation-4c, process fact), so only those
# records exist in the imputed callset. Records below it stay in the truth (they can be causal) but are unmeasured.
PANEL_MINIMUM_ALLELE_COUNT = 2


def measured_records(root: Path) -> np.ndarray:
    """Records the imputed callset contains: panel allele count in [PANEL_MINIMUM_ALLELE_COUNT, AN - that].
    Cached as measured.npy next to the cohort."""
    cached = root / "measured.npy"
    if cached.exists():
        return np.load(cached)
    panel = np.load(root / "panel_haps.npy", mmap_mode="r")
    allele_count = np.zeros(panel.shape[0], dtype=np.int64)
    for first in range(0, panel.shape[0], STREAM_ROWS):
        allele_count[first:first + STREAM_ROWS] = np.asarray(panel[first:first + STREAM_ROWS]).sum(axis=1, dtype=np.int64)
    minor = np.minimum(allele_count, panel.shape[1] - allele_count)
    measured = minor >= PANEL_MINIMUM_ALLELE_COUNT
    np.save(cached, measured)
    return measured



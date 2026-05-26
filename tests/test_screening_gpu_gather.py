"""Parity: GPU sample_intersect gather inside the screening kernel matches
the CPU rebitpack + GPU screen path that was the production fallback.

Random packed (200 samples, 100 variants), random subset of 80 sample
indices, count/sum/sumsq must match exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
screening = pytest.importorskip("sv_pgs.bitpacked.screening")

from sv_pgs.bitpacked.cpu_reference import cpu_screen  # noqa: E402
from sv_pgs.screening_pipeline import _rebitpack_chunk  # noqa: E402


def _bytes_per_variant(n_samples: int) -> int:
    return (n_samples + 3) // 4


def _rand_packed(rng: np.random.Generator, n_variants: int, n_samples: int) -> np.ndarray:
    bpv = _bytes_per_variant(n_samples)
    return rng.integers(0, 256, size=(n_variants, bpv), dtype=np.uint8)


def test_gpu_gather_matches_cpu_rebitpack():
    rng = np.random.default_rng(42)
    n_samples_raw = 200
    n_variants = 100
    n_eff = 80

    packed = _rand_packed(rng, n_variants, n_samples_raw)
    intersect = rng.choice(n_samples_raw, size=n_eff, replace=False).astype(np.int32)

    # --- CPU rebitpack baseline (existing production path) ---
    rebitpacked = _rebitpack_chunk(packed, n_samples_raw, intersect, count_a1=True)
    ref = cpu_screen(rebitpacked, n_eff, rhs=None, count_a1=True)

    # --- GPU gather path (new) ---
    packed_dev = cp.asarray(packed)
    intersect_dev = cp.asarray(intersect)
    out_count = cp.zeros(n_variants, dtype=cp.int32)
    out_sum = cp.zeros(n_variants, dtype=cp.float64)
    out_sumsq = cp.zeros(n_variants, dtype=cp.float64)
    screening.screen(
        packed_dev,
        n_eff,
        out_count,
        out_sum,
        out_sumsq,
        rhs=None,
        out_dosage_rhs=None,
        out_observed_rhs=None,
        count_a1=True,
        sample_intersect=intersect_dev,
        raw_n_samples=n_samples_raw,
    )

    np.testing.assert_array_equal(
        cp.asnumpy(out_count), ref["count"].astype(np.int32)
    )
    np.testing.assert_array_equal(cp.asnumpy(out_sum), ref["sum"])
    np.testing.assert_array_equal(cp.asnumpy(out_sumsq), ref["sumsq"])

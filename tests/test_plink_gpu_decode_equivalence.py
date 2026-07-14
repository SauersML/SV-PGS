"""Bit-exact equivalence between the GPU 2-bit decode reference and the
existing sv_pgs.plink CPU decode path.

The GPU kernel ``plink_decode_a1_subset`` matches a CPU reference
implementation in :mod:`sv_pgs.genotype`. We test the reference against
``sv_pgs.plink._BYTE_DECODE_LUT_A1`` (the production CPU decode LUT) on
randomized packed inputs and sample subsets — if the reference disagrees
with the existing decoder for any (byte, sample_offset) pair the kernel
would too, and we'd silently corrupt every fit that uses the GPU path.

No GPU required.
"""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.genotype import (
    _PLINK_DECODE_LOOKUP_A1,
    _decode_packed_bytes_reference,
)
from sv_pgs.plink import _BYTE_DECODE_LUT_A1, PLINK_MISSING_INT8


def _decode_with_production_lut(
    packed: np.ndarray,
    *,
    bytes_per_variant: int,
    sample_indices: np.ndarray,
    n_variants: int,
) -> np.ndarray:
    """Decode using the same LUT the runtime CPU path uses."""
    packed_u8 = np.asarray(packed, dtype=np.uint8).reshape(n_variants, bytes_per_variant)
    expanded = _BYTE_DECODE_LUT_A1[packed_u8].reshape(n_variants, bytes_per_variant * 4)
    expanded = expanded[:, : packed_u8.shape[1] * 4]
    sub = expanded[:, sample_indices]
    return np.asfortranarray(sub.T)


def test_lookup_lut_matches_reference() -> None:
    """The reference uses the same code->int8 mapping as the production LUT."""
    assert np.array_equal(_PLINK_DECODE_LOOKUP_A1, np.array([2, PLINK_MISSING_INT8, 1, 0], dtype=np.int8))


@pytest.mark.parametrize("n_total_samples,n_variants,n_kept", [
    (16, 4, 16),                # tiny exact-multiple-of-4
    (17, 7, 17),                # trailing partial byte
    (1000, 50, 1000),           # full sample set
    (1000, 50, 250),            # quarter sample subset
    (4096, 128, 333),           # irregular subset size
])
def test_gpu_decode_reference_matches_cpu_production(
    n_total_samples: int, n_variants: int, n_kept: int,
) -> None:
    rng = np.random.default_rng(42 + n_kept)
    bytes_per_variant = (n_total_samples + 3) // 4
    packed = rng.integers(0, 256, size=(n_variants, bytes_per_variant), dtype=np.uint8)
    sample_indices = np.sort(rng.choice(n_total_samples, size=n_kept, replace=False)).astype(np.int64)

    gpu_ref = _decode_packed_bytes_reference(
        packed,
        bytes_per_variant=bytes_per_variant,
        sample_indices=sample_indices,
        n_variants=n_variants,
    )
    cpu_prod = _decode_with_production_lut(
        packed,
        bytes_per_variant=bytes_per_variant,
        sample_indices=sample_indices,
        n_variants=n_variants,
    )
    assert gpu_ref.dtype == np.int8
    assert gpu_ref.shape == (n_kept, n_variants)
    assert np.array_equal(gpu_ref, cpu_prod), (
        f"GPU decode reference disagrees with CPU production LUT for "
        f"({n_total_samples=}, {n_variants=}, {n_kept=})"
    )


def test_all_byte_values_decode_correctly() -> None:
    """For each of 256 byte values, decode at every sample offset 0..3."""
    bytes_per_variant = 1
    # 256 variants, each with a single byte cycling through all values.
    packed = np.arange(256, dtype=np.uint8).reshape(256, 1)
    sample_indices = np.arange(4, dtype=np.int64)

    gpu_ref = _decode_packed_bytes_reference(
        packed,
        bytes_per_variant=bytes_per_variant,
        sample_indices=sample_indices,
        n_variants=256,
    )
    cpu_prod = _decode_with_production_lut(
        packed,
        bytes_per_variant=bytes_per_variant,
        sample_indices=sample_indices,
        n_variants=256,
    )
    assert np.array_equal(gpu_ref, cpu_prod)

    # Independently sanity-check that code 01 -> PLINK_MISSING_INT8, etc.
    # Construct a byte where samples 0..3 have codes 0,1,2,3.
    byte = 0b11_10_01_00  # sample 0 -> 00 (=2), sample 1 -> 01 (missing), sample 2 -> 10 (=1), sample 3 -> 11 (=0)
    packed_one = np.array([[byte]], dtype=np.uint8)
    decoded = _decode_packed_bytes_reference(
        packed_one,
        bytes_per_variant=1,
        sample_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        n_variants=1,
    )
    assert decoded[0, 0] == 2
    assert decoded[1, 0] == PLINK_MISSING_INT8
    assert decoded[2, 0] == 1
    assert decoded[3, 0] == 0


def test_sample_subset_correctness() -> None:
    """Decoding with a non-contiguous sample subset preserves order and values."""
    rng = np.random.default_rng(7)
    n_total_samples = 100
    n_variants = 5
    bytes_per_variant = (n_total_samples + 3) // 4
    packed = rng.integers(0, 256, size=(n_variants, bytes_per_variant), dtype=np.uint8)
    # Deliberately scattered, non-monotone-after-permute selection.
    sample_indices = np.array([0, 5, 17, 23, 64, 99, 31, 42], dtype=np.int64)
    # _decode_packed_bytes_reference requires sorted-or-arbitrary; production
    # path takes arbitrary too. Test with a sorted subset since the runtime
    # sample subset (RowSubsetRawGenotypeMatrix.row_indices) is sorted.
    sample_indices = np.sort(sample_indices)
    gpu_ref = _decode_packed_bytes_reference(
        packed,
        bytes_per_variant=bytes_per_variant,
        sample_indices=sample_indices,
        n_variants=n_variants,
    )
    cpu_prod = _decode_with_production_lut(
        packed,
        bytes_per_variant=bytes_per_variant,
        sample_indices=sample_indices,
        n_variants=n_variants,
    )
    assert np.array_equal(gpu_ref, cpu_prod)

"""Copy-number store columns: modal copy number, code scale, and the per-record affine decode."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.copy_number import (
    allele_count_decode,
    copy_number_codes_per_unit,
    decode_values,
    encode_copy_numbers,
    modal_copy_numbers,
)
from sv_pgs.dosage_store import CODES_PER_DOSAGE, MAXIMUM_CODE


def test_the_modal_copy_number_is_the_most_frequent_call_and_the_smallest_among_ties() -> None:
    copy_numbers = np.array([[2, 2, 3, 1, 2], [1, 3, 3, 1, 4], [0, 5, 5, 7, 7]])
    called = np.array([[True] * 5, [True] * 5, [True, True, True, False, False]])
    np.testing.assert_array_equal(modal_copy_numbers(copy_numbers, called), [2, 1, 5])
    with pytest.raises(ValueError):
        modal_copy_numbers(copy_numbers, np.zeros_like(called))


def test_every_integer_copy_number_up_to_the_maximum_is_stored_exactly() -> None:
    rng = np.random.default_rng(3)
    maximum = np.array([2, 5, 12, 127, MAXIMUM_CODE])
    copy_numbers = np.stack([rng.integers(0, top + 1, size=400) for top in maximum])
    copy_numbers[np.arange(maximum.size), 0] = maximum
    scale = copy_number_codes_per_unit(copy_numbers.max(axis=1))
    np.testing.assert_array_equal(scale, MAXIMUM_CODE // maximum)
    modal = modal_copy_numbers(copy_numbers, np.ones_like(copy_numbers, dtype=bool))
    values = decode_values(encode_copy_numbers(copy_numbers, scale), scale, -modal)
    # CN * k / k is exact in fp64 for these small integers, and so is the shift.
    np.testing.assert_array_equal(values, copy_numbers - modal[:, None])


def test_a_copy_number_past_the_code_range_is_refused() -> None:
    with pytest.raises(ValueError):
        copy_number_codes_per_unit(np.array([MAXIMUM_CODE + 1]))
    with pytest.raises(ValueError):
        encode_copy_numbers(np.array([[3, 6]]), np.array([50], dtype=np.uint8))


def test_allele_count_records_decode_as_today() -> None:
    codes = np.array([[0, 127, 254, 63]], dtype=np.uint8)
    scale, origin = allele_count_decode(1)
    assert scale[0] == CODES_PER_DOSAGE and origin[0] == 0
    np.testing.assert_array_equal(decode_values(codes, scale, origin), codes / CODES_PER_DOSAGE)

from __future__ import annotations

import numpy as np

PLINK_MISSING_INT8: np.int8 = np.int8(-127)

_DECODE_LOOKUP_A1: np.ndarray = np.array([2, PLINK_MISSING_INT8, 1, 0], dtype=np.int8)
_DECODE_LOOKUP_A2: np.ndarray = np.array([0, PLINK_MISSING_INT8, 1, 2], dtype=np.int8)


def make_decode_lut(count_a1: bool = True) -> np.ndarray:
    """Return (256, 4) int8 table mapping byte -> (s0, s1, s2, s3) dosages.
    Missing slot encoded as -127 (PLINK_MISSING_INT8, matches plink.py)."""
    per_code_lookup = _DECODE_LOOKUP_A1 if count_a1 else _DECODE_LOOKUP_A2
    lut = np.empty((256, 4), dtype=np.int8)
    for byte_value in range(256):
        for sample_offset in range(4):
            two_bit_code = (byte_value >> (2 * sample_offset)) & 0b11
            lut[byte_value, sample_offset] = per_code_lookup[two_bit_code]
    return lut

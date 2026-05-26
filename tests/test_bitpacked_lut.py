from __future__ import annotations

import numpy as np
import pytest

lut_mod = pytest.importorskip("sv_pgs.bitpacked.lut")

from sv_pgs.plink import _BYTE_DECODE_LUT_A1, _BYTE_DECODE_LUT_A2, PLINK_MISSING_INT8


def test_lut_a1_matches_plink_lut():
    lut = lut_mod.make_decode_lut(count_a1=True)
    assert np.array_equal(lut, _BYTE_DECODE_LUT_A1)


def test_lut_a2_matches_plink_lut():
    lut = lut_mod.make_decode_lut(count_a1=False)
    assert np.array_equal(lut, _BYTE_DECODE_LUT_A2)


def test_lut_shape():
    lut_a1 = lut_mod.make_decode_lut(count_a1=True)
    lut_a2 = lut_mod.make_decode_lut(count_a1=False)
    assert lut_a1.shape == (256, 4)
    assert lut_a1.dtype == np.int8
    assert lut_a2.shape == (256, 4)
    assert lut_a2.dtype == np.int8


def test_lut_missing_code_is_minus_127():
    # 0x55 = 0b01010101: all four 2-bit slots are 0b01 (missing).
    lut_a1 = lut_mod.make_decode_lut(count_a1=True)
    lut_a2 = lut_mod.make_decode_lut(count_a1=False)
    assert int(PLINK_MISSING_INT8) == -127
    assert np.array_equal(lut_a1[0x55], np.array([-127, -127, -127, -127], dtype=np.int8))
    assert np.array_equal(lut_a2[0x55], np.array([-127, -127, -127, -127], dtype=np.int8))

    # Also: every byte that contains a 0b01 slot must decode that slot to -127.
    for byte_value in range(256):
        for slot in range(4):
            two_bit_code = (byte_value >> (2 * slot)) & 0b11
            if two_bit_code == 0b01:
                assert lut_a1[byte_value, slot] == -127
                assert lut_a2[byte_value, slot] == -127


def test_lut_known_bytes():
    lut_a1 = lut_mod.make_decode_lut(count_a1=True)
    lut_a2 = lut_mod.make_decode_lut(count_a1=False)

    # A1: 00->2, 01->-127, 10->1, 11->0
    expected_a1 = {
        0x00: [2, 2, 2, 2],            # 0b00 00 00 00
        0xFF: [0, 0, 0, 0],            # 0b11 11 11 11
        0xAA: [1, 1, 1, 1],            # 0b10 10 10 10
        0x55: [-127, -127, -127, -127],  # 0b01 01 01 01
        0xE4: [2, -127, 1, 0],         # s0=00->2, s1=01->-127, s2=10->1, s3=11->0
    }
    # A2: 00->0, 01->-127, 10->1, 11->2
    expected_a2 = {
        0x00: [0, 0, 0, 0],
        0xFF: [2, 2, 2, 2],
        0xAA: [1, 1, 1, 1],
        0x55: [-127, -127, -127, -127],
        0xE4: [0, -127, 1, 2],
    }

    for byte_value, expected in expected_a1.items():
        assert np.array_equal(
            lut_a1[byte_value], np.array(expected, dtype=np.int8)
        ), f"A1 mismatch at byte 0x{byte_value:02X}"

    for byte_value, expected in expected_a2.items():
        assert np.array_equal(
            lut_a2[byte_value], np.array(expected, dtype=np.int8)
        ), f"A2 mismatch at byte 0x{byte_value:02X}"

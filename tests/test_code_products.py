"""Stage 2 code products on the CPU, against exact references."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.code_products import (
    DIGIT_BITS,
    OPERAND_DIGITS,
    CodeBlockTile,
    operand_digits,
    operand_digits_for,
    recombine_digit_products,
)


FLOAT64_ROUNDING = float(np.finfo(np.float64).eps) / 2


def _gamma(term_count: int) -> float:
    """Higham's gamma_n = n u / (1 - n u), the relative error bound of an fp64 sum of n terms."""
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    return term_count * unit_roundoff / (1.0 - term_count * unit_roundoff)


def _signed_codes(rng: np.random.Generator, variants: int, samples: int) -> np.ndarray:
    frequency = rng.uniform(0.02, 0.5, size=(variants, 1))
    dosage = rng.binomial(2, frequency, size=(variants, samples)) + rng.uniform(-0.05, 0.05, size=(variants, samples))
    codes = np.clip(np.rint(127.0 * np.clip(dosage, 0.0, 2.0)), 0, 254)
    return (codes - 127).astype(np.int8)


def _standardized(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = codes.astype(np.float64)
    means = values.mean(axis=1)
    scales = values.std(axis=1)
    return (values - means[:, None]) / scales[:, None], means, scales


def test_operand_digits_are_exact_balanced_base128_expansions() -> None:
    rng = np.random.default_rng(3)
    dense = rng.standard_normal((500, 7)) * np.exp(rng.uniform(-30, 30, 7))[None, :]
    dense[:, 3] = 0.0
    digits, scale = operand_digits(dense, np)

    assert digits.dtype == np.int8 and digits.flags.f_contiguous
    assert digits.shape == (500, OPERAND_DIGITS * 7)
    assert int(digits.min()) >= -(1 << (DIGIT_BITS - 1)) and int(digits.max()) < (1 << (DIGIT_BITS - 1))
    rebuilt = np.zeros((500, 7), dtype=np.int64)
    for digit_index in range(OPERAND_DIGITS - 1, -1, -1):
        rebuilt = rebuilt * (1 << DIGIT_BITS) + digits[:, digit_index * 7 : (digit_index + 1) * 7].astype(np.int64)
    np.testing.assert_array_equal(rebuilt, np.rint(dense * scale[None, :]).astype(np.int64))
    quantization = np.abs(rebuilt / scale[None, :] - dense).max(axis=0)
    bound = np.abs(dense).max(axis=0) * 2.0 ** -(DIGIT_BITS * OPERAND_DIGITS - 2)
    assert np.all(quantization <= bound)


def test_recombined_exact_integer_digit_products_equal_the_quantized_operand_product() -> None:
    rng = np.random.default_rng(4)
    codes = _signed_codes(rng, 40, 300)
    dense = rng.standard_normal((300, 5)) * np.exp(rng.uniform(-4, 4, 5))[None, :]
    digits, scale = operand_digits(dense, np)
    exact_products = codes.astype(np.int64) @ digits.astype(np.int64)

    recombined = recombine_digit_products(exact_products, scale, np)

    quantized = np.rint(dense * scale[None, :]) / scale[None, :]
    reference = codes.astype(np.float64) @ quantized
    # the recombination rounds once per digit, the reference GEMM once per sample
    bound = (_gamma(OPERAND_DIGITS) + _gamma(codes.shape[1])) * (np.abs(codes.astype(np.float64)) @ np.abs(quantized))
    assert np.all(np.abs(recombined - reference) <= bound)


@pytest.mark.parametrize("workspace_bytes", [1 << 34, 40_000])
def test_cpu_tile_products_match_the_dense_standardized_reference(workspace_bytes: int) -> None:
    rng = np.random.default_rng(5)
    variants, samples = 37, 1001
    codes = _signed_codes(rng, variants, samples)
    standardized, means, scales = _standardized(codes)
    tile = CodeBlockTile(codes, means, scales, np, workspace_bytes)
    right = rng.standard_normal((variants, 4))
    left = rng.standard_normal((samples, 3))
    weights = rng.uniform(0.05, 0.25, samples)
    covariates = np.column_stack([np.ones(samples), rng.standard_normal((samples, 2))])

    np.testing.assert_allclose(tile.matmat(right), standardized.T @ right, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(tile.rmatmat(left), standardized @ left, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        tile.weighted_gram(weights), standardized @ (weights[:, None] * standardized.T), rtol=1e-11, atol=1e-9
    )
    np.testing.assert_allclose(
        tile.weighted_cross(weights, covariates), standardized @ (weights[:, None] * covariates), rtol=1e-12, atol=1e-10
    )


def test_tile_rejects_codes_that_are_not_int8() -> None:
    with pytest.raises(ValueError, match="int8"):
        CodeBlockTile(np.zeros((3, 4), dtype=np.int16), np.zeros(3), np.ones(3), np, 1 << 30)


def test_a_workspace_too_small_for_one_chunk_is_refused() -> None:
    rng = np.random.default_rng(9)
    codes = _signed_codes(rng, 37, 1001)
    _standardized_codes, means, scales = _standardized(codes)
    tile = CodeBlockTile(codes, means, scales, np, 1000)
    with pytest.raises(MemoryError, match="workspace"):
        tile.rmatmat(rng.standard_normal((1001, 3)))


def _standard_tile_inputs(rng: np.random.Generator, variants: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    codes = _signed_codes(rng, variants, samples)
    standardized, means, scales = _standardized(codes)
    return codes, standardized, means, scales


def test_cpu_sample_operand_gives_the_array_rmatmat_bit_for_bit() -> None:
    rng = np.random.default_rng(11)
    codes, _standardized_codes, means, scales = _standard_tile_inputs(rng, 37, 1001)
    tiles = [CodeBlockTile(codes[rows], means[rows], scales[rows], np, 1 << 34) for rows in (slice(0, 20), slice(20, 37))]
    left = rng.standard_normal((1001, 5)) * np.exp(rng.uniform(-20, 20, 5))[None, :]
    operand = tiles[0].sample_operand(left, FLOAT64_ROUNDING)
    for tile in tiles:
        np.testing.assert_array_equal(tile.rmatmat(operand).view(np.uint64), tile.rmatmat(left).view(np.uint64))


@pytest.mark.parametrize("workspace_bytes", [1 << 34, 40_000])
def test_cpu_accumulate_matmat_adds_matmat_bit_for_bit(workspace_bytes: int) -> None:
    rng = np.random.default_rng(12)
    codes, _standardized_codes, means, scales = _standard_tile_inputs(rng, 37, 1001)
    tile = CodeBlockTile(codes, means, scales, np, workspace_bytes)
    right = rng.standard_normal((37, 4))
    expected = rng.standard_normal((1001, 4))
    fused = expected.copy()
    expected += tile.matmat(right)
    tile.accumulate_matmat(right, fused, FLOAT64_ROUNDING)
    np.testing.assert_array_equal(fused.view(np.uint64), expected.view(np.uint64))


def test_cpu_weighted_column_squares_match_the_dense_reference() -> None:
    rng = np.random.default_rng(13)
    variants, samples = 37, 1001
    codes, standardized, means, scales = _standard_tile_inputs(rng, variants, samples)
    tile = CodeBlockTile(codes, means, scales, np, 1 << 34)
    weights = rng.uniform(0.05, 0.25, (samples, 3))
    produced = tile.weighted_column_squares(weights)
    reference = np.square(standardized) @ weights
    values = codes.astype(np.float64)
    # both sides round each of their n-term sums (gamma_n) and the few terms around them (gamma_4)
    magnitude = (
        np.square(values) @ weights + 2.0 * np.abs(means)[:, None] * (np.abs(values) @ weights) + np.square(means)[:, None] * weights.sum(axis=0)[None, :]
    ) / np.square(scales)[:, None]
    bound = (_gamma(samples) + _gamma(4)) * magnitude + _gamma(samples + 2) * reference
    assert np.all(np.abs(produced - reference) <= bound)


def test_from_aligned_wraps_the_callers_codes_without_copying() -> None:
    rng = np.random.default_rng(14)
    codes, _standardized_codes, means, scales = _standard_tile_inputs(rng, 37, 1001)
    aligned = np.zeros((40, 1004), dtype=np.int8)
    aligned[:37, :1001] = codes
    wrapped = CodeBlockTile.from_aligned(aligned, 37, 1001, means, scales, float(scales.max() / scales.min()), np, 1 << 34)
    copied = CodeBlockTile(codes, means, scales, np, 1 << 34)
    left = rng.standard_normal((1001, 3))
    assert np.shares_memory(wrapped._codes, aligned)
    np.testing.assert_array_equal(wrapped.rmatmat(left).view(np.uint64), copied.rmatmat(left).view(np.uint64))
    aligned[38, 5] = 1
    aligned[3, 1002] = -4
    cleared = CodeBlockTile.from_aligned(aligned, 37, 1001, means, scales, float(scales.max() / scales.min()), np, 1 << 34)
    np.testing.assert_array_equal(cleared.rmatmat(left).view(np.uint64), copied.rmatmat(left).view(np.uint64))
    with pytest.raises(ValueError, match="aligned_codes"):
        CodeBlockTile.from_aligned(aligned[:, :1002], 37, 1001, means, scales, 1.0, np, 1 << 34)


def test_operand_digits_for_is_the_least_count_meeting_the_budget() -> None:
    rng = np.random.default_rng(15)
    values = rng.standard_normal((2000, 5)) * np.exp(rng.uniform(-8, 8, 5))[None, :]
    values[rng.random((2000, 5)) < 0.3] = 0.0
    values[:, 4] = 0.0
    for relative_error in (1e-1, 1e-4, 1e-9, FLOAT64_ROUNDING):
        count = operand_digits_for(values, relative_error, np)
        digits, scale = operand_digits(values, np, count)
        represented = recombine_digit_products(digits.astype(np.int64), scale, np)
        moved = np.linalg.norm(represented - values, axis=0)
        assert np.all(moved <= relative_error * np.linalg.norm(values, axis=0))
        live = np.linalg.norm(values, axis=0) > 0
        ratio = np.sqrt((values[:, live] != 0).sum(axis=0)) * np.abs(values[:, live]).max(axis=0) / np.linalg.norm(values[:, live], axis=0)
        # the guarantee one digit fewer would give is not enough for the worst column
        assert count == OPERAND_DIGITS or count == 1 or ratio.max() * 2.0 ** -(DIGIT_BITS * (count - 1) - 2) > relative_error
    assert operand_digits_for(values, FLOAT64_ROUNDING, np) == OPERAND_DIGITS


def test_columns_are_the_standardized_block_columns() -> None:
    rng = np.random.default_rng(16)
    codes, standardized, means, scales = _standard_tile_inputs(rng, 37, 1001)
    tile = CodeBlockTile(codes, means, scales, np, 1 << 34)
    local = np.array([0, 5, 36])
    np.testing.assert_allclose(tile.columns(local), standardized[local].T, rtol=4 * np.finfo(np.float64).eps, atol=0)

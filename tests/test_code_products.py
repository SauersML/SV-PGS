"""Stage 2 code products on the CPU, against exact references."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.code_products import (
    DIGIT_BITS,
    OPERAND_DIGITS,
    CodeBlockTile,
    operand_digits,
    recombine_digit_products,
)


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

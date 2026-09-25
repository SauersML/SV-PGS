"""Stage 2 code products on CUDA: int8 tensor-core digit GEMMs against exact references."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.code_products import DIGIT_BITS, OPERAND_DIGITS, CodeBlockTile, operand_digits, recombine_digit_products
from sv_pgs.compute_budget import _try_import_cupy
from tests.test_code_products import _signed_codes

cupy = _try_import_cupy()
FLOAT64_ROUNDING = float(np.finfo(np.float64).eps) / 2
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def test_cuda_digit_gemm_is_exact_on_integers() -> None:
    rng = np.random.default_rng(6)
    codes = _signed_codes(rng, 48, 4000)
    operand = rng.integers(-5000, 5000, size=(4000, 6)).astype(np.float64)
    tile = CodeBlockTile(cupy.asarray(codes), np.zeros(48), np.ones(48), cupy, 1 << 33)

    produced = cupy.asnumpy(tile.rmatmat(cupy.asarray(operand)))

    np.testing.assert_array_equal(produced, (codes.astype(np.int64) @ operand.astype(np.int64)).astype(np.float64))


@pytest.mark.parametrize("workspace_bytes", [1 << 33, 3_000_000])
def test_cuda_tile_products_match_the_cpu_tile(workspace_bytes: int) -> None:
    rng = np.random.default_rng(7)
    variants, samples = 61, 5003
    codes = _signed_codes(rng, variants, samples)
    values = codes.astype(np.float64)
    means, scales = values.mean(axis=1), values.std(axis=1)
    cpu_tile = CodeBlockTile(codes, means, scales, np, 1 << 34)
    cuda_tile = CodeBlockTile(cupy.asarray(codes), cupy.asarray(means), cupy.asarray(scales), cupy, workspace_bytes)
    right = rng.standard_normal((variants, 5)) * np.exp(rng.uniform(-6, 6, 5))[None, :]
    left = rng.standard_normal((samples, 3))
    weights = rng.uniform(0.05, 0.25, samples)
    covariates = np.column_stack([np.ones(samples), rng.standard_normal((samples, 2))])

    for cpu_value, cuda_value in (
        (cpu_tile.matmat(right), cuda_tile.matmat(cupy.asarray(right))),
        (cpu_tile.rmatmat(left), cuda_tile.rmatmat(cupy.asarray(left))),
        (cpu_tile.weighted_gram(weights), cuda_tile.weighted_gram(cupy.asarray(weights))),
        (cpu_tile.weighted_cross(weights, covariates), cuda_tile.weighted_cross(cupy.asarray(weights), cupy.asarray(covariates))),
    ):
        produced = cupy.asnumpy(cuda_value)
        column_error = np.abs(produced - cpu_value).max(axis=0) / np.abs(cpu_value).max(axis=0)
        assert float(column_error.max()) < 1e-10


def test_cuda_digits_match_the_numpy_digits() -> None:
    rng = np.random.default_rng(8)
    dense = rng.standard_normal((333, 4)) * np.exp(rng.uniform(-10, 10, 4))[None, :]
    cpu_digits, cpu_scale = operand_digits(dense, np)
    cuda_digits, cuda_scale = operand_digits(cupy.asarray(dense), cupy)

    np.testing.assert_array_equal(cupy.asnumpy(cuda_digits), cpu_digits)
    np.testing.assert_array_equal(cupy.asnumpy(cuda_scale), cpu_scale)
    products = cupy.asarray(rng.integers(-2**20, 2**20, size=(9, cpu_digits.shape[1])).astype(np.int32))
    np.testing.assert_allclose(
        cupy.asnumpy(recombine_digit_products(products, cuda_scale, cupy)),
        recombine_digit_products(cupy.asnumpy(products), cpu_scale, np),
        rtol=0,
        atol=0,
    )


def _cuda_tile(rng: np.random.Generator, variants: int, samples: int, workspace_bytes: int) -> tuple[CodeBlockTile, np.ndarray]:
    codes = _signed_codes(rng, variants, samples)
    values = codes.astype(np.float64)
    tile = CodeBlockTile(cupy.asarray(codes), cupy.asarray(values.mean(axis=1)), cupy.asarray(values.std(axis=1)), cupy, workspace_bytes)
    return tile, codes


def _bits(array) -> np.ndarray:
    return cupy.asnumpy(array).view(np.uint64)


@pytest.mark.parametrize(("variants", "samples", "columns"), [(37, 1003, 1), (130, 4099, 7), (61, 20011, 33)])
def test_cuda_sample_operand_gives_the_array_rmatmat_bit_for_bit(variants: int, samples: int, columns: int) -> None:
    rng = np.random.default_rng(variants + columns)
    tiles = [_cuda_tile(rng, variants, samples, 1 << 33)[0] for _ in range(3)]
    left = cupy.asarray(rng.standard_normal((samples, columns)) * np.exp(rng.uniform(-30, 30, columns))[None, :])
    operand = tiles[0].sample_operand(left, FLOAT64_ROUNDING)
    for tile in tiles:
        assert np.array_equal(_bits(tile.rmatmat(operand)), _bits(tile.rmatmat(left)))


@pytest.mark.parametrize(("variants", "samples", "columns", "workspace_bytes"), [(37, 1003, 1, 1 << 33), (130, 4099, 7, 1 << 33), (64, 9001, 5, 600_000)])
def test_cuda_accumulate_matmat_adds_matmat_bit_for_bit(variants: int, samples: int, columns: int, workspace_bytes: int) -> None:
    rng = np.random.default_rng(3 * variants + columns)
    expected = cupy.asarray(rng.standard_normal((samples, columns)))
    fused = expected.copy()
    for _ in range(3):
        tile, _codes = _cuda_tile(rng, variants, samples, workspace_bytes)
        right = cupy.asarray(rng.standard_normal((variants, columns)) * np.exp(rng.uniform(-30, 30, columns))[None, :])
        expected += tile.matmat(right)
        tile.accumulate_matmat(right, fused, FLOAT64_ROUNDING)
    assert np.array_equal(_bits(fused), _bits(expected))


def test_cuda_operand_spanning_the_int32_exact_depth_matches_exact_integers() -> None:
    rng = np.random.default_rng(15)
    samples = 270_001
    codes = _signed_codes(rng, 8, samples)
    tile = CodeBlockTile(cupy.asarray(codes), cupy.zeros(8), cupy.ones(8), cupy, 1 << 33)
    operand = rng.integers(-5000, 5000, size=(samples, 2)).astype(np.float64)
    produced = cupy.asnumpy(tile.rmatmat(tile.sample_operand(cupy.asarray(operand), FLOAT64_ROUNDING)))
    np.testing.assert_array_equal(produced, (codes.astype(np.int64) @ operand.astype(np.int64)).astype(np.float64))


def test_cuda_weighted_column_squares_match_the_cpu_tile() -> None:
    rng = np.random.default_rng(16)
    variants, samples = 61, 5003
    codes = _signed_codes(rng, variants, samples)
    values = codes.astype(np.float64)
    means, scales = values.mean(axis=1), values.std(axis=1)
    cpu_tile = CodeBlockTile(codes, means, scales, np, 1 << 34)
    cuda_tile = CodeBlockTile(cupy.asarray(codes), cupy.asarray(means), cupy.asarray(scales), cupy, 1 << 33)
    weights = rng.uniform(0.05, 0.25, (samples, 3))
    produced = cupy.asnumpy(cuda_tile.weighted_column_squares(cupy.asarray(weights)))
    expected = cpu_tile.weighted_column_squares(weights)
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    # the digits quantize each weight to 2^-(7m-2) of its column maximum; both sides then round
    # sums of at most n terms of the same magnitudes
    magnitude = (
        np.square(values) @ weights + 2.0 * np.abs(means)[:, None] * (np.abs(values) @ weights) + np.square(means)[:, None] * weights.sum(axis=0)[None, :]
    ) / np.square(scales)[:, None]
    quantization = 2.0 ** -(DIGIT_BITS * OPERAND_DIGITS - 2)
    bound = (quantization + 2 * (samples + 4) * unit_roundoff) * magnitude
    assert np.all(np.abs(produced - expected) <= bound)


@pytest.mark.parametrize("relative_error", [1e-2, 1e-5])
def test_cuda_budgeted_products_stay_within_their_bound(relative_error: float) -> None:
    rng = np.random.default_rng(17)
    variants, samples, columns = 96, 6007, 4
    codes = _signed_codes(rng, variants, samples)
    values = codes.astype(np.float64)
    means, scales = values.mean(axis=1), values.std(axis=1)
    standardized = (values - means[:, None]) / scales[:, None]
    norm = float(np.linalg.norm(standardized, 2))
    cuda_tile = CodeBlockTile(cupy.asarray(codes), cupy.asarray(means), cupy.asarray(scales), cupy, 1 << 33)
    left = rng.standard_normal((samples, columns))
    left[rng.random(samples) < 0.2] = 0.0
    operand = cuda_tile.sample_operand(cupy.asarray(left), relative_error)
    assert operand.digit_count < OPERAND_DIGITS
    produced = cupy.asnumpy(cuda_tile.rmatmat(operand))
    exact = standardized @ left
    unit_roundoff = np.finfo(np.float64).eps / 2
    # the budget moves the operand by relative_error ||L_k||; fp64 rounding adds a sum of n terms
    rounding = 2 * (samples + 4) * unit_roundoff * (np.abs(standardized) @ np.abs(left))
    assert np.all(np.linalg.norm(produced - exact, axis=0) <= relative_error * norm * np.linalg.norm(left, axis=0) + np.linalg.norm(rounding, axis=0))
    right = rng.standard_normal((variants, columns))
    image = cupy.zeros((samples, columns))
    cuda_tile.accumulate_matmat(cupy.asarray(right), image, relative_error)
    exact_image = standardized.T @ right
    rounding = 2 * (variants + 4) * unit_roundoff * (np.abs(standardized).T @ np.abs(right))
    assert np.all(np.linalg.norm(cupy.asnumpy(image) - exact_image, axis=0) <= relative_error * norm * np.linalg.norm(right, axis=0) + np.linalg.norm(rounding, axis=0))


def test_cuda_products_inside_a_tight_device_ledger_form_their_columns_by_blocks_bit_for_bit() -> None:
    """``rmatmat`` of an array and of a prepared operand, and the squared codes' product of a prepared operand, on a
    device whose ledger leaves room for their outputs and a few columns' temporaries only: the integer products are
    formed a block of columns at a time (``_column_block``), no allocation is refused, and every column is the
    unconstrained product's bit for bit (a genome fit's block CG formed ~290 columns' products at once and was refused
    1.03 GB with 41.4 GB held on an A100, bench-sim chr22 015)."""
    from sv_pgs.compute_budget import ComputeBudget
    from sv_pgs.memory_broker import memory_scope

    rng = np.random.default_rng(15)
    variants, samples, columns = 2000, 1003, 1000
    tile, codes = _cuda_tile(rng, variants, samples, 1 << 33)
    left = cupy.asarray(rng.standard_normal((samples, columns)) * np.exp(rng.uniform(-30, 30, columns))[None, :])
    operand = tile.sample_operand(left, FLOAT64_ROUNDING)
    weights = tile.sample_operand(cupy.asarray(rng.uniform(0.05, 0.25, (samples, columns))), FLOAT64_ROUNDING)
    calls = (
        (lambda: tile.rmatmat(left), 4, 0),
        (lambda: tile.rmatmat(operand), 4, 0),
        # the two digit products, their combination and sum; the int16 squares and their two int8 halves
        (lambda: tile._squared_codes_times_operand(weights), 5, 4 * codes.size),
    )
    expected = [_bits(call()) for call, _, _ in calls]
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    output_bytes = variants * columns * np.dtype(np.float64).itemsize
    # A few columns' temporaries of the widest product (the array's: its copy, padding and digit split per sample).
    few_columns = 4 * (56 * samples + 56 * variants)
    for (call, outputs, fixed), bits in zip(calls, expected):
        pool.free_all_blocks()
        held = int(pool.total_bytes())
        # The call's outputs whole and its fixed buffers, and a few columns: below the 56 bytes per variant and column
        # the products of every column at once held.
        capacity = held + outputs * output_bytes + fixed + few_columns
        budget = ComputeBudget(
            device_kind="cuda", device_ids=(0,), device_names=("ledger test",), device_bytes=(capacity,),
            device_compute_capabilities=((0, 0),), host_bytes=1 << 34, cpu_threads=1,
        )
        # Every device allocation inside the scope is admitted by the ledger's allocator, which refuses any that
        # would take the device's held bytes past the capacity: the call completing is the bound holding.
        with memory_scope(budget):
            assert tile._column_block(columns, 56 * variants) < columns
            got = call()
        assert np.array_equal(_bits(got), bits)
        del got
        pool.free_all_blocks()

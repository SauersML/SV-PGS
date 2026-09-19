"""Stage 2 code products on CUDA: int8 tensor-core digit GEMMs against exact references."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.code_products import CodeBlockTile, operand_digits, recombine_digit_products
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.genotype import _try_import_cupy
from tests.test_code_products import _cpu_budget, _signed_codes

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def _budget(device_bytes: int, host_bytes: int = 1 << 34) -> ComputeBudget:
    return ComputeBudget(
        device_kind="cuda",
        device_ids=(0,),
        device_names=("test",),
        device_bytes=(device_bytes,),
        device_compute_capabilities=((8, 0),),
        host_bytes=host_bytes,
        cpu_threads=1,
    )


def test_cuda_digit_gemm_is_exact_on_integers() -> None:
    rng = np.random.default_rng(6)
    codes = _signed_codes(rng, 48, 4000)
    operand = rng.integers(-5000, 5000, size=(4000, 6)).astype(np.float64)
    tile = CodeBlockTile(cupy.asarray(codes), np.zeros(48), np.ones(48), cupy, _budget(1 << 33))

    produced = cupy.asnumpy(tile.rmatmat(cupy.asarray(operand)))

    np.testing.assert_array_equal(produced, (codes.astype(np.int64) @ operand.astype(np.int64)).astype(np.float64))


@pytest.mark.parametrize("device_bytes", [1 << 33, 3_000_000])
def test_cuda_tile_products_match_the_cpu_tile(device_bytes: int) -> None:
    rng = np.random.default_rng(7)
    variants, samples = 61, 5003
    codes = _signed_codes(rng, variants, samples)
    values = codes.astype(np.float64)
    means, scales = values.mean(axis=1), values.std(axis=1)
    cpu_tile = CodeBlockTile(codes, means, scales, np, _cpu_budget(1 << 34))
    cuda_tile = CodeBlockTile(cupy.asarray(codes), cupy.asarray(means), cupy.asarray(scales), cupy, _budget(device_bytes))
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

"""Resident codes (``resident_codes.ResidentCodeSource``) read as the streamed per-block tiles do."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import dual_solve
from sv_pgs.code_products import CodeBlockTile
from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.resident_codes import ResidentCodeSource, read_tile_rows, resident_bytes_needed
from tests.test_dual_solve import EPS, MODEL_COUNT, _CodeTileSource, _coded_problem, _dense, _solve_bound

cupy = _try_import_cupy()
WORKSPACE_BYTES = 1 << 28
# block widths that are not multiples of the int8 GEMM alignment, so the resident rows carry alignment rows
UNALIGNED_BLOCK = 29


def _modules():
    return [np] + ([cupy] if cupy is not None else [])


def _problem(seed: int, array_module):
    standardized, codes, means, scales, _bounds, covariates, weights, variances, prior_mean, response = _coded_problem(seed)
    variant_count = codes.shape[0]
    bounds = [(start, min(start + UNALIGNED_BLOCK, variant_count)) for start in range(0, variant_count, UNALIGNED_BLOCK)]
    tiles = _CodeTileSource(codes, means, scales, bounds, array_module, WORKSPACE_BYTES)
    return standardized, tiles, bounds, covariates, weights, variances, prior_mean, response


def _resident(tiles, bounds, sample_count, sample_major: bool, workspace_bytes=WORKSPACE_BYTES):
    needed = resident_bytes_needed([stop - start for start, stop in bounds], sample_count)
    return ResidentCodeSource(tiles, (2 if sample_major else 1) * needed, workspace_bytes)


def _smallest_workspace(sample_count: int, columns: int) -> int:
    low, high = 1, WORKSPACE_BYTES
    while low < high:
        middle = (low + high) // 2
        try:
            read_tile_rows(sample_count, columns, middle)
            high = middle
        except MemoryError:
            low = middle + 1
    return low


def _host(values):
    return cupy.asnumpy(values) if cupy is not None and not isinstance(values, np.ndarray) else values


@pytest.mark.parametrize("array_module", _modules(), ids=lambda module: module.__name__)
def test_resident_blocks_give_the_streamed_tiles_products(array_module) -> None:
    standardized, tiles, bounds, *_rest = _problem(3, array_module)
    resident = _resident(tiles, bounds, standardized.shape[0], sample_major=True)
    generator = np.random.default_rng(4)
    left = array_module.asarray(generator.standard_normal((standardized.shape[0], 5)))
    for (start, stop, tile), (_index, streamed) in zip(resident.blocks(), tiles.iter_tiles()):
        right = array_module.asarray(generator.standard_normal((stop - start, 5)))
        # the integer products and their recombination are the same operations on the same codes
        assert np.array_equal(_host(tile.rmatmat(left)), _host(streamed.rmatmat(left)))
        assert np.array_equal(_host(tile.matmat(right)), _host(streamed.matmat(right)))


@pytest.mark.parametrize("array_module", _modules(), ids=lambda module: module.__name__)
def test_a_read_over_resident_tiles_gives_the_dense_operator(array_module) -> None:
    standardized, tiles, bounds, covariates, weights, variances, _prior_mean, _response = _problem(5, array_module)
    sample_count = standardized.shape[0]
    # the smallest workspace a read fits: many small read tiles, each of whose products must fit it
    columns = MODEL_COUNT
    resident = _resident(tiles, bounds, sample_count, sample_major=False, workspace_bytes=_smallest_workspace(sample_count, columns))
    assert len(resident.read_tiles(columns)) > 1
    models = dual_solve.DualModels(array_module.asarray(weights), array_module.asarray(variances), array_module.asarray(covariates), array_module)
    values = np.random.default_rng(6).standard_normal((sample_count, columns))
    column_models = array_module.arange(MODEL_COUNT)
    image = _host(dual_solve.apply_operator(resident, models, array_module.asarray(values), column_models, 0.0, dual_solve.PassCount(), "exact"))
    for model in range(MODEL_COUNT):
        operator = _dense(standardized, covariates, weights, variances, model)[3]
        expected = operator @ values[:, model]
        # exact products: the two differ by the rounding of S v, n eps ||S|| ||v||
        assert np.linalg.norm(image[:, model] - expected) <= sample_count * EPS * np.linalg.norm(operator, 2) * np.linalg.norm(values[:, model])


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
def test_cuda_sample_major_codes_give_the_transposed_products_bit_for_bit() -> None:
    standardized, tiles, bounds, covariates, weights, variances, _prior_mean, _response = _problem(7, cupy)
    sample_count = standardized.shape[0]
    held = _resident(tiles, bounds, sample_count, sample_major=True)
    transposed = _resident(tiles, bounds, sample_count, sample_major=False)
    assert held.sample_major and not transposed.sample_major
    models = dual_solve.DualModels(cupy.asarray(weights), cupy.asarray(variances), cupy.asarray(covariates), cupy)
    values = cupy.asarray(np.random.default_rng(8).standard_normal((sample_count, MODEL_COUNT)))
    for relative_error in (0.0, 2.0**-14):
        images = [
            _host(dual_solve.apply_operator(source, models, values, cupy.arange(MODEL_COUNT), relative_error, dual_solve.PassCount(), "read"))
            for source in (held, transposed)
        ]
        assert np.array_equal(images[0], images[1])


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
def test_cuda_resident_codes_with_relaxed_digits_keep_the_certificate() -> None:
    standardized, tiles, bounds, covariates, weights, variances, prior_mean, response = _problem(43, cupy)
    source = _resident(tiles, bounds, standardized.shape[0], sample_major=True)
    models = dual_solve.DualModels(cupy.asarray(weights), cupy.asarray(variances), cupy.asarray(covariates), cupy)
    right = dual_solve.mean_right_hand_side(models, cupy.asarray(response), cupy.asarray(standardized @ prior_mean))
    count = dual_solve.PassCount()
    deflation, _resolved = dual_solve.spike_deflation(source, models, count)
    host_right = cupy.asnumpy(right)
    bound = cupy.asarray(_solve_bound(host_right))
    result = dual_solve.certified_block_cg(source, models, right, cupy.zeros_like(right), cupy.arange(MODEL_COUNT), bound, count, deflation=deflation)
    assert bool(cupy.all(result.residual_norm <= bound))
    assert any(relative_error > 0.0 for relative_error in result.relative_errors)
    exact = host_right - np.column_stack([
        _dense(standardized, covariates, weights, variances, model)[3] @ cupy.asnumpy(result.solution[:, model]) for model in range(MODEL_COUNT)
    ])
    assert np.all(
        np.linalg.norm(exact, axis=0)
        <= cupy.asnumpy(bound) * (1.0 + standardized.shape[0] * EPS) + standardized.shape[0] * EPS * np.linalg.norm(host_right, axis=0)
    )


def test_resident_codes_that_do_not_fit_the_plan_are_refused() -> None:
    standardized, tiles, bounds, *_rest = _problem(9, np)
    needed = resident_bytes_needed([stop - start for start, stop in bounds], standardized.shape[0])
    with pytest.raises(MemoryError):
        ResidentCodeSource(tiles, needed - 1, WORKSPACE_BYTES)


def test_a_tile_rejects_a_sample_major_array_that_does_not_hold_it() -> None:
    codes = np.zeros((8, 12), dtype=np.int8)
    with pytest.raises(ValueError):
        CodeBlockTile.from_aligned(codes, 8, 12, np.zeros(8), np.ones(8), None, np, WORKSPACE_BYTES, (np.zeros((12, 6), dtype=np.int8), 0))

"""Several CUDA devices give one device's answer: Stage 0 bit for bit, Stage 2 within its certificate."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sv_pgs import dual_solve
from sv_pgs.compute_budget import _cupy_device_context, _try_import_cupy
from sv_pgs.device_plan import ShardedDualSource, plan_blocks, reassociation_bound

cupy = _try_import_cupy()
DEVICE_COUNT = 0 if cupy is None else int(cupy.cuda.runtime.getDeviceCount())
pytestmark = pytest.mark.skipif(DEVICE_COUNT < 2, reason="needs two CUDA devices")

WORKSPACE_BYTES = 1 << 28


class _ShardTiles:
    """One device's code tiles, built on that device, streamed in its planned order."""

    def __init__(self, codes, means, scales, bounds, blocks):
        from sv_pgs.code_products import CodeBlockTile

        self._tiles = [
            CodeBlockTile(cupy.asarray(codes[bounds[block][0] : bounds[block][1]]), means[bounds[block][0] : bounds[block][1]],
                          scales[bounds[block][0] : bounds[block][1]], cupy, WORKSPACE_BYTES)
            for block in blocks
        ]

    def iter_tiles(self):
        yield from enumerate(self._tiles)


def _sharded(codes, means, scales, bounds, device_ids):
    plan = plan_blocks(device_ids, [stop - start for start, stop in bounds], int(codes.shape[1]))
    shards = []
    for position, device_id in enumerate(device_ids):
        with _cupy_device_context(cupy, device_id):
            shards.append(_ShardTiles(codes, means, scales, bounds, plan.blocks_on(position)))
    return ShardedDualSource(shards, plan, bounds, int(codes.shape[1]), cupy)


def test_a_sharded_source_solves_like_one_device() -> None:
    from tests.test_dual_solve import EPS, MODEL_COUNT, _CodeTileSource, _coded_problem, _solve_bound

    standardized, codes, means, scales, bounds, covariates, weights, variances, prior_mean, response = _coded_problem(43)
    solutions = []
    with _cupy_device_context(cupy, 0):
        single = dual_solve.StreamedDualSource(_CodeTileSource(codes, means, scales, bounds, cupy, WORKSPACE_BYTES))
        sharded = _sharded(codes, means, scales, bounds, (0, 1))
        for source in (single, sharded):
            models = dual_solve.DualModels(cupy.asarray(weights), cupy.asarray(variances), cupy.asarray(covariates), cupy)
            right = dual_solve.mean_right_hand_side(models, cupy.asarray(response), cupy.asarray(standardized @ prior_mean))
            count = dual_solve.PassCount()
            bound = cupy.asarray(_solve_bound(cupy.asnumpy(right)))
            result = dual_solve.certified_block_cg(source, models, right, cupy.zeros_like(right), cupy.arange(MODEL_COUNT), bound, count)
            assert bool(cupy.all(result.residual_norm <= bound))
            solutions.append((cupy.asnumpy(result.solution), cupy.asnumpy(bound)))
    (one, one_bound), (two, _two_bound) = solutions
    assert np.all(np.linalg.norm(one - two, axis=0) <= 2.0 * one_bound * (1.0 + standardized.shape[0] * EPS))


def test_map_reduce_sums_every_block_within_the_reassociation_bound() -> None:
    from tests.test_dual_solve import _coded_problem

    standardized, codes, means, scales, bounds, _covariates, _weights, variances, _prior_mean, response = _coded_problem(47)

    def work(start, stop, tile, shared, rows, image):
        image += tile.matmat(rows["variances"] * tile.rmatmat(shared["left"]))

    with _cupy_device_context(cupy, 0):
        left = cupy.asarray(response)
        device_variances = cupy.asarray(variances)
        sharded = _sharded(codes, means, scales, bounds, (0, 1))
        total = sharded.map_reduce(work, {"left": left}, {"variances": device_variances}, tuple(left.shape))
        one_device = _sharded(codes, means, scales, bounds, (0,))
        block_images = []
        for start, stop, tile in one_device.blocks():
            block_images.append(tile.matmat(device_variances[start:stop] * tile.rmatmat(left)))
        sequential = cupy.zeros_like(left)
        for image in block_images:
            sequential += image
        assert bool(cupy.all(cupy.abs(total - sequential) <= reassociation_bound(block_images, cupy)))


def test_stage0_writes_the_same_bits_on_one_and_two_devices(tmp_path) -> None:
    from sv_pgs.compute_budget import detect_compute_budget
    from sv_pgs.config import ModelConfig
    from sv_pgs.genotype_statistics import compute_genotype_statistics
    from tests.test_genotype_statistics import BLOCK_CAP, _statistics_dataset

    budget = detect_compute_budget()
    one = replace(
        budget, device_ids=budget.device_ids[:1], device_names=budget.device_names[:1],
        device_bytes=budget.device_bytes[:1], device_compute_capabilities=budget.device_compute_capabilities[:1],
    )
    two = replace(
        budget, device_ids=budget.device_ids[:2], device_names=budget.device_names[:2],
        device_bytes=budget.device_bytes[:2], device_compute_capabilities=budget.device_compute_capabilities[:2],
    )
    config = ModelConfig(minimum_minor_allele_frequency=0.01)
    results = []
    for label, device_budget in (("one", one), ("two", two)):
        source, training, covariates, targets = _statistics_dataset(21)
        results.append(compute_genotype_statistics(source, training, covariates, targets, config, device_budget, BLOCK_CAP, tmp_path / label))
    first, second = results
    np.testing.assert_array_equal(first.active_rows, second.active_rows)
    np.testing.assert_array_equal(first.ld.block_boundaries, second.ld.block_boundaries)
    for block_index in range(first.ld.block_count):
        left, right = first.ld.block(block_index), second.ld.block(block_index)
        np.testing.assert_array_equal(left.projected_gram, right.projected_gram)
        np.testing.assert_array_equal(left.projected_score, right.projected_score)
        np.testing.assert_array_equal(left.covariate_cross, right.covariate_cross)
        adjacent_left, adjacent_right = first.ld.adjacent_block(block_index), second.ld.adjacent_block(block_index)
        assert (adjacent_left is None) == (adjacent_right is None)
        if adjacent_left is not None:
            np.testing.assert_array_equal(adjacent_left, adjacent_right)

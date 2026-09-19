"""The CUDA Stage 0 backend reproduces the CPU backend bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import genotype_buffers
from sv_pgs.genotype_buffers import CudaGenotypeBuffer, HostGenotypeBuffer, build_sample_layout
from sv_pgs.genotype_statistics import plan_genotype_pass, run_genotype_pass
from tests.stage0_support import InMemoryTileSource, bubble_groups, mosaic_codes

cp = pytest.importorskip("cupy")

SAMPLES = 700
BLOCK_CAP = 128


def _source(seed: int, variant_counts: dict[str, int]) -> InMemoryTileSource:
    rng = np.random.default_rng(seed)
    return InMemoryTileSource(
        codes={name: mosaic_codes(rng, SAMPLES, count) for name, count in variant_counts.items()},
        groups={name: bubble_groups(rng, count) for name, count in variant_counts.items()},
    )


def _run(source, sample_groups, backend_kind, columns, devices=1):
    layout = build_sample_layout(sample_groups, profile_target=300)
    plan = plan_genotype_pass(BLOCK_CAP)
    if backend_kind == "cpu":
        buffers = [HostGenotypeBuffer(layout, plan.capacity_rows, columns, worker_count=2)]
    else:
        buffers = [
            CudaGenotypeBuffer(cp, index % cp.cuda.runtime.getDeviceCount(), layout, plan.capacity_rows,
                               plan.tile_rows, columns)
            for index in range(devices)
        ]
    blocks = []
    summary = run_genotype_pass(source, layout, buffers, plan, lambda block, buffer: blocks.append(block.on_host(buffer)))
    return summary, sorted(blocks, key=lambda block: (block.chromosome, block.start))


@pytest.fixture(scope="module")
def dataset():
    source = _source(21, {"chr4": 900, "chr9": 333})
    labels = np.random.default_rng(22).integers(0, 2, size=SAMPLES)
    labels[::13] = -1
    columns = np.random.default_rng(23).normal(size=(SAMPLES, 3))
    return source, labels, columns


def test_cuda_blocks_equal_cpu_blocks(dataset) -> None:
    source, labels, columns = dataset
    cpu_summary, cpu_blocks = _run(source, labels, "cpu", columns)
    cuda_summary, cuda_blocks = _run(source, labels, "cuda", columns)
    for name in ("chr4", "chr9"):
        np.testing.assert_array_equal(cpu_summary.chromosomes[name].cut_costs, cuda_summary.chromosomes[name].cut_costs)
        np.testing.assert_array_equal(cpu_summary.chromosomes[name].boundaries, cuda_summary.chromosomes[name].boundaries)
    assert len(cpu_blocks) == len(cuda_blocks)
    signed = {name: codes.astype(np.int64) - 127 for name, codes in source.codes.items()}
    for cpu_block, cuda_block in zip(cpu_blocks, cuda_blocks):
        assert (cpu_block.chromosome, cpu_block.start, cpu_block.stop) == (
            cuda_block.chromosome, cuda_block.start, cuda_block.stop)
        assert cuda_block.grams.dtype == np.int32
        np.testing.assert_array_equal(cpu_block.grams, cuda_block.grams)
        np.testing.assert_array_equal(cpu_block.sums, cuda_block.sums)
        np.testing.assert_allclose(cpu_block.cross_products, cuda_block.cross_products, rtol=1e-12, atol=1e-9)
        members = np.flatnonzero(labels == 1)
        values = signed[cuda_block.chromosome][cuda_block.start : cuda_block.stop, members]
        np.testing.assert_array_equal(cuda_block.grams[1], values @ values.T)


def test_long_sample_ranges_accumulate_exactly_in_int64(dataset, monkeypatch) -> None:
    source, labels, columns = dataset
    monkeypatch.setattr(genotype_buffers, "_CUDA_SAMPLE_CHUNK", 128)
    _, cuda_blocks = _run(source, labels, "cuda", None)
    signed = {name: codes.astype(np.int64) - 127 for name, codes in source.codes.items()}
    members = np.flatnonzero(labels == 0)
    for block in cuda_blocks:
        values = signed[block.chromosome][block.start : block.stop, members]
        np.testing.assert_array_equal(block.grams[0], values @ values.T)


def test_two_device_workers_equal_one(dataset) -> None:
    source, labels, _ = dataset
    _, single = _run(source, labels, "cuda", None, devices=1)
    _, double = _run(source, labels, "cuda", None, devices=2)
    assert [(block.chromosome, block.start) for block in single] == [(block.chromosome, block.start) for block in double]
    for left, right in zip(single, double):
        np.testing.assert_array_equal(left.grams, right.grams)


def test_cuda_rejects_missing_codes_only_for_included_samples() -> None:
    source = _source(24, {"chr7": 200})
    labels = np.zeros(SAMPLES, dtype=np.int64)
    labels[0] = -1
    source.codes["chr7"][49, 0] = 255
    _run(source, labels, "cuda", None)
    source.codes["chr7"][50, 1] = 255
    with pytest.raises(ValueError, match="missing"):
        _run(source, labels, "cuda", None)


def test_cuda_projected_ld_equals_cpu(tmp_path) -> None:
    from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
    from sv_pgs.config import ModelConfig
    from sv_pgs.genotype_statistics import compute_genotype_statistics

    source = _source(31, {"chr2": 500, "chr3": 300})
    source.codes["chr2"][21] = source.codes["chr2"][20]
    source.codes["chr2"][22] = 254 - source.codes["chr2"][20]
    rng = np.random.default_rng(32)
    training = np.sort(rng.choice(SAMPLES, size=SAMPLES - 50, replace=False))
    covariates = np.column_stack([np.ones(training.shape[0]), rng.normal(size=(training.shape[0], 3))])
    targets = rng.normal(size=(training.shape[0], 2))
    cpu_budget = ComputeBudget(device_kind="cpu", device_ids=(), device_names=(), device_bytes=(),
                               device_compute_capabilities=(), host_bytes=8 * 10**9, cpu_threads=2)
    results = [
        compute_genotype_statistics(source, training, covariates, targets, ModelConfig(), budget, BLOCK_CAP,
                                    tmp_path / name)
        for name, budget in (("cpu", cpu_budget), ("cuda", detect_compute_budget()))
    ]
    cpu, cuda = results
    np.testing.assert_array_equal(cpu.active_rows, cuda.active_rows)
    np.testing.assert_array_equal(cpu.tie_map.kept_indices, cuda.tie_map.kept_indices)
    np.testing.assert_array_equal(cpu.tie_map.original_to_reduced, cuda.tie_map.original_to_reduced)
    np.testing.assert_array_equal(cpu.boundaries.block_starts, cuda.boundaries.block_starts)
    np.testing.assert_array_equal(cpu.means, cuda.means)
    np.testing.assert_allclose(cpu.scales, cuda.scales, rtol=1e-15)
    for block_index in range(cpu.ld.block_count):
        left, right = cpu.ld.block(block_index), cuda.ld.block(block_index)
        np.testing.assert_array_equal(left.reduced_columns, right.reduced_columns)
        np.testing.assert_allclose(left.projected_gram, right.projected_gram, rtol=1e-6, atol=1e-6 * SAMPLES)
        np.testing.assert_allclose(left.projected_score, right.projected_score, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(left.covariate_cross, right.covariate_cross, rtol=1e-10, atol=1e-8)

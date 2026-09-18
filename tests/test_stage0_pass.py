"""The Stage 0 pass on the CPU backend: exact integer statistics and the optimal LD partition."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.stage0 import (
    CpuStage0Backend,
    block_correlation,
    build_sample_layout,
    centered_cross_products,
    column_moments,
    cut_allowed_from_groups,
    plan_genotype_pass,
    run_genotype_pass,
)
from sv_pgs.stage0.genotype_pass import assign_chromosomes
from tests.stage0_support import (
    InMemoryTileSource,
    bubble_groups,
    mosaic_codes,
    reference_cut_costs,
    reference_cuts,
)

SAMPLES = 540
BLOCK_CAP = 128


def _dataset(seed: int, variant_counts: dict[str, int]) -> InMemoryTileSource:
    rng = np.random.default_rng(seed)
    codes = {name: mosaic_codes(rng, SAMPLES, count) for name, count in variant_counts.items()}
    groups = {name: bubble_groups(rng, count) for name, count in variant_counts.items()}
    return InMemoryTileSource(codes=codes, groups=groups)


def _sample_groups(seed: int) -> np.ndarray:
    labels = np.random.default_rng(seed).integers(0, 3, size=SAMPLES)
    labels[::17] = -1
    return labels


def _run(source: InMemoryTileSource, sample_groups: np.ndarray, devices: int, columns: np.ndarray | None,
         profile_target: int = 16384):
    layout = build_sample_layout(sample_groups, profile_target=profile_target)
    plan = plan_genotype_pass(BLOCK_CAP)
    backends = [CpuStage0Backend(layout, plan.capacity_rows, columns, worker_count=3) for _ in range(devices)]
    blocks = []
    summary = run_genotype_pass(source, layout, backends, plan, blocks.append)
    for backend in backends:
        backend.close()
    return layout, summary, blocks


def test_block_statistics_are_exact_and_blocks_tile_each_chromosome() -> None:
    source = _dataset(1, {"chr21": 700})
    sample_groups = _sample_groups(2)
    columns = np.random.default_rng(3).normal(size=(SAMPLES, 2))
    layout, summary, blocks = _run(source, sample_groups, devices=1, columns=columns)
    codes = source.codes["chr21"]
    boundaries = summary.chromosomes["chr21"].boundaries
    assert boundaries[0] == 0 and boundaries[-1] == codes.shape[0]
    assert np.all(np.diff(boundaries) <= BLOCK_CAP)
    assert np.all(cut_allowed_from_groups(source.groups["chr21"])[boundaries])
    assert [(block.start, block.stop) for block in blocks] == list(zip(boundaries[:-1], boundaries[1:]))
    signed = codes.astype(np.int64) - 127
    for block in blocks:
        assert block.grams.dtype == np.int32
        for group in range(3):
            members = np.flatnonzero(sample_groups == group)
            values = signed[block.start : block.stop, members]
            np.testing.assert_array_equal(block.grams[group], values @ values.T)
            np.testing.assert_array_equal(block.sums[group], values.sum(axis=1))
            assert block.group_counts[group] == members.shape[0]
            np.testing.assert_allclose(block.cross_products[group], values @ columns[members], rtol=1e-13, atol=1e-9)


def test_correlation_matches_a_float64_reference() -> None:
    source = _dataset(4, {"chr22": 400})
    sample_groups = _sample_groups(5)
    columns = np.random.default_rng(6).normal(size=(SAMPLES, 3))
    _, _, blocks = _run(source, sample_groups, devices=1, columns=columns)
    dosage = source.codes["chr22"].astype(np.float64) / 127.0
    members = np.flatnonzero(np.isin(sample_groups, [0, 2]))
    for block in blocks:
        values = dosage[block.start : block.stop, members].T
        mean = values.mean(axis=0)
        deviation = values.std(axis=0)
        varying = deviation > 0
        standardized = np.zeros_like(values)
        standardized[:, varying] = (values[:, varying] - mean[varying]) / deviation[varying]
        count, correlation, constant = block_correlation(block, [0, 2])
        assert count == members.shape[0]
        np.testing.assert_array_equal(constant, ~varying)
        np.testing.assert_allclose(correlation, standardized.T @ standardized / count, rtol=0, atol=1e-12)
        fitted_mean, fitted_deviation = column_moments(block, [0, 2])
        np.testing.assert_allclose(fitted_mean, mean, rtol=1e-13)
        np.testing.assert_allclose(fitted_deviation, deviation, rtol=1e-12, atol=1e-15)
        column_sums = np.stack([columns[sample_groups == group].sum(axis=0) for group in range(3)])
        centered = columns[members] - columns[members].mean(axis=0)
        np.testing.assert_allclose(
            centered_cross_products(block, [0, 2], column_sums), standardized.T @ centered, rtol=1e-10, atol=1e-9
        )


@pytest.mark.parametrize("profile_target", [16384, 200])
def test_partition_is_the_exact_optimum_of_the_cut_costs(profile_target: int) -> None:
    source = _dataset(7, {"chr20": 900})
    sample_groups = _sample_groups(8)
    layout, summary, _ = _run(source, sample_groups, devices=1, columns=None, profile_target=profile_target)
    assert layout.profile_count <= profile_target
    result = summary.chromosomes["chr20"]
    assert result.forced_cuts == 0
    costs = reference_cut_costs(source.codes["chr20"], layout, BLOCK_CAP)
    expected = reference_cuts(costs, cut_allowed_from_groups(source.groups["chr20"]), BLOCK_CAP)
    assert result.boundaries.tolist() == expected


def test_several_devices_give_the_single_device_blocks() -> None:
    source = _dataset(9, {"chr1": 520, "chr2": 300, "chr3": 450})
    sample_groups = _sample_groups(10)
    _, single_summary, single = _run(source, sample_groups, devices=1, columns=None)
    _, multi_summary, multi = _run(source, sample_groups, devices=2, columns=None)
    key = lambda block: (block.chromosome, block.start)
    single = sorted(single, key=key)
    multi = sorted(multi, key=key)
    assert [key(block) for block in single] == [key(block) for block in multi]
    for left, right in zip(single, multi):
        np.testing.assert_array_equal(left.grams, right.grams)
        np.testing.assert_array_equal(left.sums, right.sums)
    for name in ("chr1", "chr2", "chr3"):
        np.testing.assert_array_equal(
            single_summary.chromosomes[name].boundaries, multi_summary.chromosomes[name].boundaries
        )


def test_missing_codes_are_rejected() -> None:
    source = _dataset(11, {"chr5": 200})
    source.codes["chr5"][37, 5] = 255
    with pytest.raises(ValueError, match="missing"):
        _run(source, _sample_groups(12), devices=1, columns=None)


def test_chromosomes_go_longest_first_to_the_least_loaded_device() -> None:
    assignment = assign_chromosomes({"a": 5, "b": 9, "c": 4, "d": 3}, 2)
    assert assignment == [["b", "d"], ["a", "c"]]

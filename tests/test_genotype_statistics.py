"""The Stage 0 pass on the CPU backend: exact integer statistics and the optimal LD partition."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.genotype_buffers import HostGenotypeBuffer, build_sample_layout
from sv_pgs.genotype_statistics import (
    TIE_CORRELATION_SCREEN,
    assign_chromosomes,
    plan_genotype_pass,
    run_cross_product_pass,
    run_genotype_pass,
)
from sv_pgs.ld_partition import cut_allowed_from_groups
from tests.stage0_support import (
    InMemoryTileSource,
    bubble_groups,
    mosaic_codes,
    reference_cut_costs,
    reference_cuts,
)

SAMPLES = 540
BLOCK_CAP = 128


def _dataset(seed: int, variant_counts: dict[str, int], regular_hotspots: bool = False) -> InMemoryTileSource:
    rng = np.random.default_rng(seed)
    spacing = 100 if regular_hotspots else 60
    codes = {
        name: mosaic_codes(rng, SAMPLES, count, hotspot_spacing=spacing, regular_hotspots=regular_hotspots)
        for name, count in variant_counts.items()
    }
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
    backends = [HostGenotypeBuffer(layout, plan.capacity_rows, columns, worker_count=3) for _ in range(devices)]
    blocks = []
    summary = run_genotype_pass(source, layout, backends, plan, lambda block, buffer: blocks.append(block))
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
    for position, block in enumerate(blocks):
        assert block.grams.dtype == np.int32
        if position == 0:
            assert block.previous_start is None and block.previous_grams is None
        else:
            assert block.previous_start == blocks[position - 1].start
            assert block.previous_grams.dtype == np.int32
        for group in range(3):
            members = np.flatnonzero(sample_groups == group)
            values = signed[block.start : block.stop, members]
            np.testing.assert_array_equal(block.grams[group], values @ values.T)
            if position:
                previous = signed[block.previous_start : block.start, members]
                np.testing.assert_array_equal(block.previous_grams[group], previous @ values.T)
            np.testing.assert_array_equal(block.sums[group], values.sum(axis=1))
            assert block.group_counts[group] == members.shape[0]
            np.testing.assert_allclose(block.cross_products[group], values @ columns[members], rtol=1e-13, atol=1e-9)


def test_group_statistics_add_up_to_any_union_of_groups() -> None:
    source = _dataset(4, {"chr22": 400})
    sample_groups = _sample_groups(5)
    _, _, blocks = _run(source, sample_groups, devices=1, columns=None)
    signed = source.codes["chr22"].astype(np.int64) - 127
    members = np.flatnonzero(np.isin(sample_groups, [0, 2]))
    for block in blocks:
        values = signed[block.start : block.stop, members]
        np.testing.assert_array_equal(block.grams[[0, 2]].astype(np.int64).sum(axis=0), values @ values.T)
        np.testing.assert_array_equal(block.sums[[0, 2]].sum(axis=0), values.sum(axis=1))
        assert block.group_counts[[0, 2]].sum() == members.shape[0]


@pytest.mark.parametrize("profile_target", [16384, 200])
def test_partition_is_the_exact_optimum_of_the_cut_costs(profile_target: int) -> None:
    source = _dataset(7, {"chr20": 900}, regular_hotspots=True)
    sample_groups = _sample_groups(8)
    layout, summary, _ = _run(source, sample_groups, devices=1, columns=None, profile_target=profile_target)
    assert layout.profile_count <= profile_target
    result = summary.chromosomes["chr20"]
    costs = reference_cut_costs(source.codes["chr20"], layout, BLOCK_CAP)
    np.testing.assert_array_equal(result.cut_costs, costs)
    assert result.forced_cuts == 0
    expected = reference_cuts(costs, cut_allowed_from_groups(source.groups["chr20"]), BLOCK_CAP)
    assert result.boundaries.tolist() == expected


def test_a_full_buffer_forces_valid_cuts_on_ambiguous_ld() -> None:
    source = _dataset(7, {"chr20": 900})
    sample_groups = _sample_groups(8)
    layout, summary, blocks = _run(source, sample_groups, devices=1, columns=None)
    result = summary.chromosomes["chr20"]
    np.testing.assert_array_equal(result.cut_costs, reference_cut_costs(source.codes["chr20"], layout, BLOCK_CAP))
    assert result.forced_cuts > 0
    assert np.all(np.diff(result.boundaries) <= BLOCK_CAP)
    assert np.all(cut_allowed_from_groups(source.groups["chr20"])[result.boundaries])
    signed = source.codes["chr20"].astype(np.int64) - 127
    members = np.flatnonzero(sample_groups == 1)
    for block in blocks:
        values = signed[block.start : block.stop, members]
        np.testing.assert_array_equal(block.grams[1], values @ values.T)


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


def test_missing_codes_are_rejected_only_for_included_samples() -> None:
    source = _dataset(11, {"chr5": 200})
    sample_groups = _sample_groups(12)
    assert sample_groups[0] == -1
    source.codes["chr5"][36, 0] = 255
    _run(source, sample_groups, devices=1, columns=None)
    source.codes["chr5"][37, 5] = 255
    with pytest.raises(ValueError, match="missing"):
        _run(source, sample_groups, devices=1, columns=None)


def test_chromosomes_go_longest_first_to_the_least_loaded_device() -> None:
    assignment = assign_chromosomes({"a": 5, "b": 9, "c": 4, "d": 3}, 2)
    assert assignment == [["b", "d"], ["a", "c"]]


def test_a_later_cross_product_pass_matches_the_fused_one() -> None:
    source = _dataset(13, {"chr6": 350, "chr8": 200})
    sample_groups = _sample_groups(14)
    columns = np.random.default_rng(15).normal(size=(SAMPLES, 4))
    layout = build_sample_layout(sample_groups)
    buffers = [HostGenotypeBuffer(layout, 100, columns, worker_count=2) for _ in range(2)]
    tiles = {}
    run_cross_product_pass(source, buffers, 100, lambda name, start, stop, products: tiles.update(
        {(name, start): (stop, products)}))
    signed = {name: codes.astype(np.int64) - 127 for name, codes in source.codes.items()}
    for (name, start), (stop, products) in tiles.items():
        for group in range(3):
            members = np.flatnonzero(sample_groups == group)
            np.testing.assert_allclose(
                products[group], signed[name][start:stop, members] @ columns[members], rtol=1e-12, atol=1e-9
            )
    assert sorted(tiles) == [("chr6", 0), ("chr6", 100), ("chr6", 200), ("chr6", 300), ("chr8", 0), ("chr8", 100)]


def _cpu_budget():
    from sv_pgs.compute_budget import ComputeBudget

    return ComputeBudget(device_kind="cpu", device_ids=(), device_names=(), device_bytes=(),
                         device_compute_capabilities=(), host_bytes=8 * 10**9, cpu_threads=2)


def _statistics_dataset(seed: int):
    source = _dataset(seed, {"chr2": 420, "chr3": 260})
    codes = source.codes["chr2"]
    codes[11] = codes[10]
    codes[12] = 254 - codes[10]
    codes[40] = codes[10] // 2
    codes[41] = codes[40] + 127
    codes[70, :] = 0
    codes[70, :3] = 127
    rng = np.random.default_rng(seed + 1)
    training = np.sort(rng.choice(SAMPLES, size=SAMPLES - 60, replace=False))
    covariates = np.column_stack([np.ones(training.shape[0]), rng.normal(size=(training.shape[0], 2))])
    targets = rng.normal(size=(training.shape[0], 2))
    return source, training, covariates, targets


def test_projected_ld_matches_a_dense_float64_reference(tmp_path) -> None:
    from sv_pgs.config import ModelConfig
    from sv_pgs.genotype_statistics import compute_genotype_statistics

    source, training, covariates, targets = _statistics_dataset(21)
    config = ModelConfig(minimum_minor_allele_frequency=0.01)
    statistics = compute_genotype_statistics(
        source, training, covariates, targets, config, _cpu_budget(), BLOCK_CAP, tmp_path / "ld"
    )
    all_codes = np.concatenate([source.codes["chr2"], source.codes["chr3"]])
    dosage = all_codes[:, training].astype(np.float64).T / 127.0
    frequency = dosage.mean(axis=0) / 2.0
    deviation = dosage.std(axis=0)
    expected_active = (np.minimum(frequency, 1 - frequency) >= 0.01) & (deviation >= config.minimum_scale) & (np.ptp(dosage, axis=0) > 0)
    np.testing.assert_array_equal(statistics.active_rows, np.flatnonzero(expected_active))
    np.testing.assert_allclose(statistics.allele_frequency, frequency[expected_active], rtol=1e-12)
    active_dosage = dosage[:, expected_active]
    standardized = (active_dosage - active_dosage.mean(axis=0)) / active_dosage.std(axis=0)
    np.testing.assert_allclose(statistics.scales / 127.0, active_dosage.std(axis=0), rtol=1e-12)

    tie_map = statistics.tie_map
    kept = tie_map.kept_indices
    assert kept.shape[0] == statistics.block_of_reduced.shape[0]
    active_position = {row: index for index, row in enumerate(statistics.active_rows.tolist())}
    assert tie_map.original_to_reduced[active_position[11]] == tie_map.original_to_reduced[active_position[10]]
    assert tie_map.original_to_reduced[active_position[12]] == tie_map.original_to_reduced[active_position[10]]
    assert tie_map.original_to_reduced[active_position[41]] == tie_map.original_to_reduced[active_position[40]]
    group = tie_map.reduced_to_group[tie_map.original_to_reduced[active_position[10]]]
    assert group.representative_index == active_position[10]
    np.testing.assert_array_equal(group.signs, [1.0, 1.0, -1.0])
    for index in range(standardized.shape[1]):
        reduced = tie_map.original_to_reduced[index]
        representative = kept[reduced]
        correlation = standardized[:, index] @ standardized[:, representative] / standardized.shape[0]
        assert abs(abs(correlation) - 1.0) < 1e-12 or representative == index

    reduced_x = standardized[:, kept]
    hat = covariates @ np.linalg.solve(covariates.T @ covariates, covariates.T)
    projected_x = reduced_x - hat @ reduced_x
    projected_y = targets - hat @ targets
    np.testing.assert_allclose(statistics.covariate_gram, covariates.T @ covariates, rtol=1e-12)
    np.testing.assert_allclose(statistics.target_gram, targets.T @ targets, rtol=1e-12)
    for block_index in range(statistics.ld.block_count):
        block = statistics.ld.block(block_index)
        columns = block.reduced_columns
        assert np.all(statistics.block_of_reduced[columns] == block_index)
        reference_gram = projected_x[:, columns].T @ projected_x[:, columns]
        np.testing.assert_allclose(block.projected_gram, reference_gram, rtol=2e-6, atol=2e-6 * standardized.shape[0])
        np.testing.assert_allclose(block.projected_score, projected_x[:, columns].T @ projected_y, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(block.covariate_cross, reduced_x[:, columns].T @ covariates, rtol=1e-9, atol=1e-8)
    ld = statistics.ld
    assert ld.block_boundaries[-1] == kept.shape[0]
    correlation = projected_x.T @ projected_x / standardized.shape[0]
    np.testing.assert_allclose(ld.ld_diagonal(), np.diagonal(correlation), rtol=2e-6)
    reference_scores = np.concatenate([
        (correlation[start:stop, start:stop] ** 2).sum(axis=0)
        for start, stop in zip(ld.block_boundaries[:-1], ld.block_boundaries[1:])
    ])
    np.testing.assert_allclose(ld.ld_scores(), reference_scores, rtol=2e-5)
    for block_index in range(ld.block_count):
        block_correlation = ld.correlation_block(block_index)
        assert block_correlation.dtype == np.float32
        np.testing.assert_array_equal(block_correlation, block_correlation.T)
    # the adjacent blocks' coupling: every intermediate is a sum of at most n + k terms of
    # standardized products, each at most n in magnitude, then one float32 rounding of the result
    sample_count, covariate_count = standardized.shape[0], covariates.shape[1]
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    fp64_bound = 4 * (sample_count + covariate_count) * unit_roundoff * sample_count
    firsts = 0
    for block_index in range(ld.block_count):
        adjacent = ld.adjacent_block(block_index)
        block = ld.block(block_index)
        if block_index == 0 or ld.block(block_index - 1).chromosome != block.chromosome:
            assert adjacent is None
            firsts += 1
            continue
        reference = projected_x[:, ld.block(block_index - 1).reduced_columns].T @ projected_x[:, block.reduced_columns]
        assert adjacent.dtype == np.float32
        bound = np.finfo(np.float32).eps / 2.0 * np.abs(reference) + fp64_bound
        assert np.all(np.abs(adjacent.astype(np.float64) - reference) <= bound)
    assert firsts == 2
    boundaries = statistics.boundaries
    assert boundaries.block_count == statistics.ld.block_count
    assert len(boundaries.signature_sha256()) == 64


class _ArrayStore:
    """The DosageCodeStore protocol over in-memory codes."""

    def __init__(self, codes_by_chromosome: dict, group_first) -> None:
        self.chromosomes = tuple(codes_by_chromosome)
        counts = [codes.shape[0] for codes in codes_by_chromosome.values()]
        self.chromosome_starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._codes = np.concatenate(list(codes_by_chromosome.values()))
        self.variant_table = type("VariantTable", (), {"group_first": group_first})()
        self.reads = 0

    @property
    def n_samples(self) -> int:
        return self._codes.shape[1]

    @property
    def n_variants(self) -> int:
        return self._codes.shape[0]

    def read_codes(self, start, stop, sample_indices=None, out=None):
        self.reads += 1
        out[...] = self._codes[start:stop]
        return out


def test_the_dosage_store_source_streams_the_candidate_rows() -> None:
    from sv_pgs.genotype_statistics import DosageStoreTileSource

    source = _dataset(17, {"chr1": 400, "chr2": 300})
    all_codes = np.concatenate([source.codes["chr1"], source.codes["chr2"]])
    group_first = np.concatenate([source.groups["chr1"], 400 + source.groups["chr2"]])
    group_first = np.maximum.accumulate(np.searchsorted(group_first, group_first))
    store = _ArrayStore(dict(source.codes), group_first)
    rng = np.random.default_rng(18)
    candidates = np.sort(rng.choice(700, size=520, replace=False))
    tile_source = DosageStoreTileSource(store, candidates)
    assert tile_source.chromosomes() == ["chr1", "chr2"]
    rows = tile_source.store_rows("chr2")
    np.testing.assert_array_equal(rows, candidates[candidates >= 400])
    out = np.empty((rows.shape[0], SAMPLES), dtype=np.uint8)
    tile_source.read_rows("chr2", 0, rows.shape[0], out)
    np.testing.assert_array_equal(out, all_codes[rows])
    np.testing.assert_array_equal(tile_source.unsplittable_groups("chr2"), group_first[rows])
    layout, summary, blocks = _run(tile_source, _sample_groups(19), devices=1, columns=None)
    signed = all_codes.astype(np.int64) - 127
    members = np.flatnonzero(_sample_groups(19) == 0)
    for block in blocks:
        store_rows = tile_source.store_rows(block.chromosome)[block.start : block.stop]
        values = signed[store_rows][:, members]
        np.testing.assert_array_equal(block.grams[0], values @ values.T)


def test_the_tie_screen_keeps_every_exact_tie_under_fp64_rounding():
    # The correlation is formed as in _project_block: exact integer N, then two inverse roots.
    random_generator = np.random.default_rng(13)
    count = 50_000
    lowest = 1.0
    for _ in range(2000):
        base = random_generator.integers(-40, 41, size=count)
        shift, slope = int(random_generator.integers(-40, 41)), int(random_generator.choice([-2, -1, 1, 2]))
        tied = shift + slope * base
        codes = np.stack([base, tied]).astype(np.int64)
        sums = codes.sum(axis=1).astype(np.float64)
        gram = (codes @ codes.T).astype(np.float64)
        numerator = count * np.diag(gram) - sums * sums
        inverse_root = 1.0 / np.sqrt(numerator)
        correlation = (count * gram[0, 1] - sums[0] * sums[1]) * inverse_root[0] * inverse_root[1]
        lowest = min(lowest, abs(float(correlation)))
    assert lowest >= TIE_CORRELATION_SCREEN

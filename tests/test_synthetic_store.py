from pathlib import Path

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import VariantClass
from sv_pgs.dosage_store import VARIANT_CLASSES, DosageStore, dosage_array_directory, signed_code_moments
from sv_pgs.synthetic_store import (
    ANCESTRIES,
    CLASS_LEGEND,
    ERR_IMP,
    NOISE_CLASSES,
    POP_KEPT_PATHS,
    RECORD_NESTED,
    RECORD_PATH,
    RECORD_SINGLE,
    HaplotypeSource,
    NoiseParameters,
    block_boundaries,
    draw_tile_mosaic,
    generate_block,
    generate_store,
    haplotype_posteriors,
    haplotype_r2,
    plan_store,
    pop_bubble,
    solve_error_rate,
)


def _source(
    variant_count: int = 600,
    founders_per_ancestry: int = 40,
    seed: int = 4,
    rare_fraction: float = 0.6,
) -> HaplotypeSource:
    """Two source chromosomes of founder haplotypes with blocky LD: runs of variants copy their
    run's first variant half the time."""
    rng = np.random.default_rng(seed)
    haplotype_count = 2 * founders_per_ancestry * len(ANCESTRIES)
    frequencies = np.where(
        rng.random(variant_count) < rare_fraction,
        rng.uniform(0.005, 0.02, variant_count),
        rng.uniform(0.05, 0.5, variant_count),
    )
    haplotypes = (rng.random((variant_count, haplotype_count)) < frequencies[:, None]).astype(np.uint8)
    for start in range(0, variant_count - 1, 10):
        haplotypes[start + 1 : start + 4] = np.where(rng.random((3, 1)) < 0.5, haplotypes[start], haplotypes[start + 1 : start + 4])
    ancestry = np.repeat(np.arange(len(ANCESTRIES)), 2 * founders_per_ancestry).astype(np.int8)
    classes = rng.choice(len(CLASS_LEGEND), size=variant_count, p=[0.8, 0.15, 0.05]).astype(np.uint8)
    split = variant_count // 2
    positions = np.concatenate(
        [np.sort(rng.choice(np.arange(1, 4_000_000), size=count, replace=False)) for count in (split, variant_count - split)]
    )
    return HaplotypeSource(
        positions=positions.astype(np.int64),
        genetic_map_cm=positions / 1e6,
        reference_lengths=np.ones(variant_count, dtype=np.int64),
        alternate_lengths=np.ones(variant_count, dtype=np.int64),
        class_codes=classes,
        variant_classes=np.minimum(classes, 1).astype(np.uint8),
        tandem_repeat=rng.random(variant_count) < 0.5,
        chromosome_starts=np.array([0, split, variant_count]),
        haplotypes=haplotypes,
        haplotype_ancestry=ancestry,
        ancestry_frequencies=np.stack(
            [haplotypes[:, ancestry == index].mean(axis=1) for index in range(len(ANCESTRIES))], axis=1
        ).astype(np.float32),
    )


def _plan(root: Path, source: HaplotypeSource, records: int = 1500, samples: tuple[int, ...] = (60, 40)):
    return plan_store(
        root,
        source,
        half_sample_counts=samples,
        half_pipelines=("A", "B")[: len(samples)],
        total_records=records,
        chromosome_count=2,
        seed=17,
        block_records=96,
        codec="raw",
        shard_rows=256,
        inner_rows=32,
    )


def test_generated_store_is_deterministic_across_worker_counts_and_matches_its_sidecar(tmp_path: Path) -> None:
    source = _source()
    serial = _plan(tmp_path / "serial", source)
    generate_store(serial, workers=1)
    parallel = _plan(tmp_path / "parallel", source)
    generate_store(parallel, workers=3)
    for half_index in range(2):
        for chromosome in serial.chromosomes:
            shard_directory = dosage_array_directory(tmp_path / "serial", half_index, chromosome) / "c"
            serial_files = sorted(path for path in shard_directory.rglob("0") if path.is_file())
            assert serial_files
            for serial_file in serial_files:
                parallel_file = tmp_path / "parallel" / serial_file.relative_to(tmp_path / "serial")
                assert serial_file.read_bytes() == parallel_file.read_bytes()
    budget = ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(),
        host_bytes=1 << 26, cpu_threads=2,
    )
    with DosageStore.open(tmp_path / "parallel") as store:
        assert (store.n_variants, store.n_samples) == (1500, 100)
        moments = signed_code_moments(store, 0, store.n_variants, np.arange(store.n_samples), budget)
        assert moments.sample_count == 100
        for name in ("sum_code", "sum_ds", "n_off_mode_code", "ds_mode_milli"):
            assert store.statistic(name).shape == (1500,)
        kinds = np.concatenate([layout.record_kind for layout in parallel.layouts])
        assert {RECORD_SINGLE, RECORD_PATH, RECORD_NESTED} <= set(kinds.tolist())
        table = store.variant_table
        assert np.array_equal(table.annotations["n_paths_total"] > 1, kinds != RECORD_SINGLE)
        assert np.all(np.diff(table.position[table.chromosome == 1]) >= 0)
        assert np.all(np.diff(table.genetic_position_cm[table.chromosome == 1]) >= 0)
        offsets = np.cumsum([0] + [layout.record_count for layout in parallel.layouts[:-1]])
        bubble_starts = np.concatenate([layout.bubble_start + offset for layout, offset in zip(parallel.layouts, offsets)])
        assert np.array_equal(table.group_first, bubble_starts)


def test_err_imp_floor_and_pop_normalization_set_the_background_code(tmp_path: Path) -> None:
    source = _source(variant_count=800, rare_fraction=0.97)
    plan = _plan(tmp_path / "store", source, records=4000, samples=(400,))
    generate_store(plan, workers=1)
    with DosageStore.open(tmp_path / "store") as store:
        mode = store.statistic("ds_mode_milli")
        annotations = store.variant_table.annotations
        has_pl = annotations["has_pl"].astype(bool)
        frequency = annotations["panel_af"]
        total_paths = annotations["n_paths_total"].astype(int)
        carried = annotations["n_paths"].astype(int)
    kinds = np.concatenate([layout.record_kind for layout in plan.layouts])
    snv = np.concatenate([layout.noise_class for layout in plan.layouts]) == NOISE_CLASSES.index("SNV")
    rare = frequency < 0.03
    single = (kinds == RECORD_SINGLE) & rare & snv
    assert np.all(mode[single & ~has_pl] == 2)
    assert np.all(mode[single & has_pl] == 0)
    assert np.any(single & has_pl) and np.any(single & ~has_pl)
    odds = ERR_IMP / (1 - ERR_IMP)
    bubble_starts = np.concatenate([layout.bubble_start + offset for layout, offset in zip(plan.layouts, (0, plan.layouts[0].record_count))])
    rare_snv_paths = np.array(
        [bool(np.all((rare & snv)[start : start + paths])) for start, paths in zip(bubble_starts, total_paths)]
    )
    nested = (kinds == RECORD_NESTED) & (total_paths <= POP_KEPT_PATHS) & rare_snv_paths
    assert nested.any()
    expected = np.rint(2000 * carried[nested] * odds / (1 + total_paths[nested] * odds))
    assert np.array_equal(mode[nested], expected)
    assert np.any(expected >= 4)


def test_pop_bubble_matches_a_per_sample_rule_a_reference() -> None:
    rng = np.random.default_rng(8)
    path_count, sample_count = POP_KEPT_PATHS + 4, 30
    posteriors = np.clip(rng.beta(0.3, 3.0, size=(path_count, 2 * sample_count)), 1e-5, 1 - 1e-5).astype(np.float32)
    mask = 0b1011
    popped = pop_bubble(posteriors.copy(), [mask])
    for sample in range(sample_count):
        pair = posteriors[:, 2 * sample : 2 * sample + 2].astype(np.float64)
        kept = np.argsort(-(pair[:, 0] + pair[:, 1]), kind="stable")[:POP_KEPT_PATHS]
        for haplotype in range(2):
            odds = np.zeros(path_count)
            odds[kept] = pair[kept, haplotype] / (1 - pair[kept, haplotype])
            expected = odds / (1 + odds.sum())
            column = 2 * sample + haplotype
            assert np.allclose(popped[:path_count, column], expected, rtol=1e-5, atol=1e-7)
            assert np.isclose(popped[path_count, column], expected[[0, 1, 3]].sum(), rtol=1e-5, atol=1e-7)


def test_mixture_r2_formula_and_solver_match_simulated_posteriors() -> None:
    rng = np.random.default_rng(12)
    frequency = np.array([0.02, 0.1, 0.35])
    target = np.array([0.6, 0.9, 0.97])
    uninformed = 0.5 * (1 - target)
    soft = np.minimum(0.1, 1 - uninformed)
    error_rate = solve_error_rate(target, uninformed, soft, frequency)
    assert np.allclose(haplotype_r2(uninformed, error_rate, soft, frequency), target, atol=1e-9)
    haplotypes = 400_000
    truth = (rng.random((3, haplotypes)) < frequency[:, None]).astype(np.uint8)
    carrier_flip = np.minimum(error_rate / (2 * frequency), 1 - uninformed - soft)
    noncarrier_flip = np.minimum(error_rate / (2 * (1 - frequency)), 1 - uninformed - soft)
    parameters = NoiseParameters(
        uninformed=uninformed.astype(np.float32),
        carrier_flip=carrier_flip.astype(np.float32),
        noncarrier_flip=noncarrier_flip.astype(np.float32),
        soft=soft.astype(np.float32),
        floor=np.full(3, ERR_IMP, dtype=np.float32),
    )
    ancestry_frequency = np.repeat(frequency[:, None], len(ANCESTRIES), axis=1).astype(np.float32)
    posteriors = haplotype_posteriors(truth, parameters, ancestry_frequency, np.zeros(haplotypes, dtype=np.int8), rng)
    realized = [np.corrcoef(posteriors[row], truth[row])[0, 1] ** 2 for row in range(3)]
    assert np.allclose(realized, target, atol=0.01)


def test_single_path_dosage_r2_tracks_the_per_class_target(tmp_path: Path) -> None:
    source = _source(variant_count=3000, founders_per_ancestry=60)
    plan = _plan(tmp_path / "store", source, records=2500, samples=(3000,))
    layout = plan.layouts[0]
    mosaic = draw_tile_mosaic(plan.source, int(layout.tile[0]), plan.cohort, np.random.default_rng(2))
    block_start, block_stop = block_boundaries(layout, 0, layout.record_count, 1500)[0]
    truth, popped = generate_block(plan, 0, block_start, block_stop, mosaic)
    dosage = popped[0][:, 0::2] + popped[0][:, 1::2]
    genotype = truth[:, 0::2].astype(np.float64) + truth[:, 1::2]
    records = slice(block_start, block_stop)
    snv_common = (
        (layout.record_kind[records] == RECORD_SINGLE)
        & (layout.noise_class[records] == NOISE_CLASSES.index("SNV"))
        & (source.pooled_frequencies[layout.source_index[records]] > 0.05)
    )
    assert snv_common.sum() >= 20
    r2 = [np.corrcoef(dosage[row], genotype[row])[0, 1] ** 2 for row in np.flatnonzero(snv_common) if genotype[row].std() > 0]
    assert 0.93 < float(np.mean(r2)) < 1.0


@pytest.mark.parametrize("block_records", [7, 50, 1000])
def test_blocks_never_split_bubbles_or_tiles(tmp_path: Path, block_records: int) -> None:
    source = _source(variant_count=300)
    plan = _plan(tmp_path / "store", source, records=1500)
    layout = plan.layouts[0]
    blocks = block_boundaries(layout, 0, layout.record_count, block_records)
    assert blocks[0][0] == 0 and blocks[-1][1] == layout.record_count
    assert all(left[1] == right[0] for left, right in zip(blocks, blocks[1:]))
    for start, stop in blocks:
        assert layout.bubble_start[start] == start
        assert np.all(layout.bubble_start[start:stop] >= start)
        assert len(set(layout.tile[start:stop].tolist())) == 1


def test_source_loader_reads_design_credit_archives_and_maps_classes(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    founders = 12
    superpopulation = np.array(ANCESTRIES * 3)[:founders]
    kinds = np.array(["SNV", "INDEL", "DEL", "DEL_SEQ", "DUP", "INS:ME:ALU", "INS_SEQ", "INV", "DEL"])
    sv_length = np.array([0, 3, 2000, 60, 300, 280, 90, 5000, 400])
    in_tr = np.array([False, True, False, False, False, False, False, False, True])
    paths = []
    for chromosome in (21, 22):
        haplotypes = (rng.random((kinds.size, 2 * founders)) < 0.3).astype(np.uint8)
        path = tmp_path / f"src_chr{chromosome}.npz"
        np.savez(
            path,
            positions=np.arange(kinds.size) * 100 + 5,
            cm=np.arange(kinds.size) * 0.01,
            ref_len=np.ones(kinds.size, dtype=np.int64),
            alt_len=np.ones(kinds.size, dtype=np.int64),
            sv_length=sv_length,
            kinds=kinds,
            in_tr=in_tr,
            superpop=superpopulation,
            n_samples=np.int64(founders),
            packed_haps=np.packbits(haplotypes, axis=1),
        )
        paths.append(path)
    source = HaplotypeSource.load(paths)
    assert source.haplotypes.shape == (2 * kinds.size, 2 * founders)
    assert source.tile_range(0) == (0, kinds.size) and source.tile_range(3) == (kinds.size, 2 * kinds.size)
    expected = [
        VariantClass.SNV, VariantClass.SMALL_INDEL, VariantClass.DELETION_LONG, VariantClass.DELETION_SHORT,
        VariantClass.DUPLICATION_SHORT, VariantClass.INSERTION_MEI, VariantClass.OTHER_COMPLEX_SV,
        VariantClass.INVERSION_BND_COMPLEX, VariantClass.STR_VNTR_REPEAT,
    ]
    assert [VARIANT_CLASSES[code] for code in source.variant_classes[: kinds.size]] == expected
    assert source.class_codes[: kinds.size].tolist() == [0, 1, 2, 2, 2, 2, 2, 2, 2]

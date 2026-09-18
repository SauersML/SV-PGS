"""Cross-source SV fusion: matching rules, two-source calibration, fused column."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.sv_fusion import SvSites, calibrate_two_sources, candidate_pairs, fused_dosage


def _sites(rows: list[tuple[str, int, int, int, str]], duplications_are_insertions: bool) -> SvSites:
    chromosomes, starts, ends, sizes, kinds = zip(*rows)
    return SvSites(
        chromosomes=np.asarray(chromosomes),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        sizes=np.asarray(sizes, dtype=np.int64),
        kinds=np.asarray(kinds),
        duplications_are_insertions=duplications_are_insertions,
    )


# First source: a sequence-resolved panel (DUP = inserted copy, a point).
_PANEL = _sites(
    [
        ("chr1", 10_000, 12_000, 2_000, "DEL"),  # 0: overlaps gatksv 0 by 1600/2000
        ("chr1", 20_000, 21_000, 1_000, "DEL"),  # 1: gatksv 1 is 3x longer (size ratio fails)
        ("chr1", 30_050, 30_051, 800, "INS"),  # 2: 50 bp from gatksv 2 (INS)
        ("chr1", 40_000, 40_001, 1_500, "INS"),  # 3: at the END of gatksv 3 (DUP interval)
        ("chr1", 50_000, 50_001, 600, "DUP"),  # 4: sequence-resolved DUP vs gatksv 4 DUP interval start
        ("chr1", 60_000, 61_000, 1_000, "DEL"),  # 5: gatksv 5 is an INS (incompatible kinds)
        ("chr2", 10_000, 12_000, 2_000, "DEL"),  # 6: other chromosome than gatksv 0
        ("chr1", 70_200, 70_201, 900, "INS"),  # 7: 200 bp from gatksv 6 (too far)
    ],
    duplications_are_insertions=True,
)
# Second source: GATK-SV (DUP = the duplicated interval).
_GATKSV = _sites(
    [
        ("chr1", 10_400, 12_400, 2_000, "DEL"),  # 0
        ("chr1", 20_000, 23_000, 3_000, "DEL"),  # 1
        ("chr1", 30_000, 30_001, 900, "INS"),  # 2
        ("chr1", 38_500, 40_020, 1_520, "DUP"),  # 3
        ("chr1", 49_990, 50_590, 600, "DUP"),  # 4
        ("chr1", 60_000, 60_001, 1_000, "INS"),  # 5
        ("chr1", 70_000, 70_001, 900, "INS"),  # 6
    ],
    duplications_are_insertions=False,
)


def test_candidate_pairs_follow_the_gatksv_clustering_rules() -> None:
    pairs = candidate_pairs(_PANEL, _GATKSV)

    assert list(zip(pairs.first_rows.tolist(), pairs.second_rows.tolist())) == [(0, 0), (2, 2), (3, 3), (4, 4)]
    np.testing.assert_allclose(pairs.reciprocal_overlaps[0], 0.8)
    assert np.isnan(pairs.reciprocal_overlaps[1:]).all()
    assert pairs.breakpoint_distances.tolist() == [-1, 50, 20, 10]
    np.testing.assert_allclose(pairs.size_ratios, [1.0, 800 / 900, 1500 / 1520, 1.0])


def test_duplication_geometry_depends_on_the_source_representation() -> None:
    # Two GATK-SV-style DUP intervals are compared by reciprocal overlap.
    first = _sites([("chr1", 1_000, 3_000, 2_000, "DUP")], duplications_are_insertions=False)
    second = _sites([("chr1", 1_500, 3_500, 2_000, "DUP")], duplications_are_insertions=False)

    pairs = candidate_pairs(first, second)

    np.testing.assert_allclose(pairs.reciprocal_overlaps, [0.75])
    assert pairs.breakpoint_distances.tolist() == [-1]


# ---------------------------------------------------------------------------
# Two-source calibration, on simulations with a known genotype
# ---------------------------------------------------------------------------


def _squared_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean()
    second = second - second.mean()
    return float((first @ second) ** 2 / ((first @ first) * (second @ second)))


def _two_sources(
    generator: np.random.Generator,
    allele_frequencies: np.ndarray,
    separation: float,
    miss_rate: float,
    false_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genotype g, a calibrated imputed DS, and hard calls with missed and false alleles."""
    frequency = allele_frequencies[:, None]
    haplotypes = generator.random((allele_frequencies.shape[0], 2)) < frequency
    evidence = generator.standard_normal(haplotypes.shape) + separation * haplotypes
    log_odds = separation * evidence - separation**2 / 2 + np.log(frequency / (1 - frequency))
    dosage = (1 / (1 + np.exp(-log_odds))).sum(axis=1)
    called = np.where(
        haplotypes,
        generator.random(haplotypes.shape) > miss_rate,
        generator.random(haplotypes.shape) < false_rate,
    )
    return haplotypes.sum(axis=1).astype(np.float64), dosage, called.sum(axis=1).astype(np.float64)


def test_calibration_recovers_both_reliabilities_and_fusion_beats_each_source() -> None:
    generator = np.random.default_rng(3)
    sample_count = 80_000
    genotype, dosage, hard_calls = _two_sources(generator, np.full(sample_count, 0.05), 2.0, 0.2, 0.002)
    observed = generator.random(sample_count) > 0.1
    groups = np.zeros(sample_count, dtype=np.int64)

    calibration = calibrate_two_sources(dosage, hard_calls, observed, groups)
    fused = fused_dosage(calibration, dosage, hard_calls, observed)

    assert calibration.accepted
    assert abs(calibration.first_reliability - _squared_correlation(dosage[observed], genotype[observed])) < 0.03
    assert abs(calibration.second_reliability - _squared_correlation(hard_calls[observed], genotype[observed])) < 0.03
    fused_truth = _squared_correlation(fused[observed], genotype[observed])
    assert fused_truth > _squared_correlation(dosage[observed], genotype[observed]) + 0.3
    assert fused_truth > _squared_correlation(hard_calls[observed], genotype[observed])
    assert abs(calibration.fused_reliability - fused_truth) < 0.03
    # A no-call in the second source keeps the imputed dosage.
    np.testing.assert_array_equal(fused[~observed], dosage[~observed])


def test_unrelated_records_fail_the_calibration_check() -> None:
    generator = np.random.default_rng(5)
    sample_count = 50_000
    _, dosage, _ = _two_sources(generator, np.full(sample_count, 0.1), 2.5, 0.1, 0.001)
    _, _, unrelated_calls = _two_sources(generator, np.full(sample_count, 0.1), 2.5, 0.1, 0.001)
    observed = np.ones(sample_count, dtype=bool)

    calibration = calibrate_two_sources(dosage, unrelated_calls, observed, np.zeros(sample_count, dtype=np.int64))

    assert not calibration.accepted
    with pytest.raises(ValueError, match="keep them as separate columns"):
        fused_dosage(calibration, dosage, unrelated_calls, observed)


def test_group_labels_correct_the_genotype_variance_of_an_admixed_sample() -> None:
    generator = np.random.default_rng(11)
    sample_count = 80_000
    groups = (generator.random(sample_count) < 0.5).astype(np.int64)
    frequencies = np.where(groups == 1, 0.35, 0.02)
    genotype, dosage, hard_calls = _two_sources(generator, frequencies, 2.5, 0.2, 0.002)
    observed = np.ones(sample_count, dtype=bool)
    true_first_reliability = _squared_correlation(dosage, genotype)

    grouped = calibrate_two_sources(dosage, hard_calls, observed, groups)
    pooled = calibrate_two_sources(dosage, hard_calls, observed, np.zeros(sample_count, dtype=np.int64))

    assert abs(grouped.first_reliability - true_first_reliability) < 0.03
    assert abs(pooled.first_reliability - true_first_reliability) > 3 * abs(grouped.first_reliability - true_first_reliability)

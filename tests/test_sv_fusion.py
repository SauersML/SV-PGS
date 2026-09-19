"""Cross-source SV fusion: matching rules, two-source calibration, fused column."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.sv_fusion import (
    SvCandidatePairs,
    SvSites,
    TwoSourceCalibration,
    calibrate_two_sources,
    candidate_pairs,
    fused_dosage,
    resolve_one_to_one,
    sequence_resolved_sites,
)


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
    odds_deflation: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genotype g, an imputed DS with its posterior variance, and hard calls with missed and false alleles.

    Each haplotype's posterior uses its sample's own allele frequency, as a
    haplotype-panel imputer's does, so the DS is calibrated; an
    ``odds_deflation`` below 1 shrinks every posterior log-odds toward 0 and
    makes it underconfident.
    """
    frequency = allele_frequencies[:, None]
    haplotypes = generator.random((allele_frequencies.shape[0], 2)) < frequency
    evidence = generator.standard_normal(haplotypes.shape) + separation * haplotypes
    log_odds = separation * evidence - separation**2 / 2 + np.log(frequency / (1 - frequency))
    posteriors = 1 / (1 + np.exp(-odds_deflation * log_odds))
    called = np.where(
        haplotypes,
        generator.random(haplotypes.shape) > miss_rate,
        generator.random(haplotypes.shape) < false_rate,
    )
    return (
        haplotypes.sum(axis=1).astype(np.float64),
        posteriors.sum(axis=1),
        (posteriors * (1 - posteriors)).sum(axis=1),
        called.sum(axis=1).astype(np.float64),
    )


def test_calibration_recovers_both_reliabilities_and_fusion_beats_each_source() -> None:
    generator = np.random.default_rng(3)
    sample_count = 80_000
    genotype, dosage, posterior_variance, hard_calls = _two_sources(generator, np.full(sample_count, 0.05), 2.0, 0.2, 0.002)
    observed = generator.random(sample_count) > 0.1

    calibration = calibrate_two_sources(dosage, posterior_variance, hard_calls, observed)
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
    _, dosage, posterior_variance, _ = _two_sources(generator, np.full(sample_count, 0.1), 2.5, 0.1, 0.001)
    _, _, _, unrelated_calls = _two_sources(generator, np.full(sample_count, 0.1), 2.5, 0.1, 0.001)
    observed = np.ones(sample_count, dtype=bool)

    calibration = calibrate_two_sources(dosage, posterior_variance, unrelated_calls, observed)

    assert not calibration.accepted
    with pytest.raises(ValueError, match="keep them as separate columns"):
        fused_dosage(calibration, dosage, unrelated_calls, observed)


def test_structured_sample_needs_no_ancestry_labels() -> None:
    # Two groups with very different frequencies: a pooled HWE variance would
    # be far off, but the posterior's own variance gives Var(g) exactly.
    generator = np.random.default_rng(11)
    sample_count = 80_000
    frequencies = np.where(generator.random(sample_count) < 0.5, 0.35, 0.02)
    genotype, dosage, posterior_variance, hard_calls = _two_sources(generator, frequencies, 2.5, 0.2, 0.002)
    observed = np.ones(sample_count, dtype=bool)

    calibration = calibrate_two_sources(dosage, posterior_variance, hard_calls, observed)

    assert calibration.accepted
    assert abs(calibration.genotype_variance - genotype.var()) < 0.02 * genotype.var()
    assert abs(calibration.first_reliability - _squared_correlation(dosage, genotype)) < 0.02
    assert abs(calibration.second_reliability - _squared_correlation(hard_calls, genotype)) < 0.02


def test_an_underconfident_dosage_fails_the_calibration_check() -> None:
    # Deflated posterior odds claim more uncertainty than the DS has, so the
    # second source correlates with it more than its claimed r2 allows.
    generator = np.random.default_rng(13)
    sample_count = 40_000
    _, dosage, posterior_variance, hard_calls = _two_sources(
        generator, np.full(sample_count, 0.2), 3.0, 0.05, 0.001, odds_deflation=0.3
    )

    calibration = calibrate_two_sources(dosage, posterior_variance, hard_calls, np.ones(sample_count, dtype=bool))

    assert calibration.pairing_z >= 5 and calibration.first_reliability <= 1.0
    assert calibration.second_reliability > 1.0
    assert not calibration.accepted


def _calibration(pairing_z: float, accepted: bool) -> TwoSourceCalibration:
    reliability = 0.8 if accepted else 1.5
    return TwoSourceCalibration(
        sample_count=1000,
        first_mean=0.1,
        second_mean=0.1,
        genotype_variance=0.1,
        second_slope=0.9,
        first_reliability=reliability,
        second_reliability=0.7,
        fused_reliability=0.9,
        first_weight=0.5,
        second_weight=0.5,
        pairing_z=pairing_z,
    )


def test_one_to_one_resolution_keeps_the_strongest_accepted_pairs() -> None:
    # Candidates (first, second): two panel alleles compete for GATK-SV record
    # 0; panel allele 2 matches two GATK-SV records; one pair fails the check.
    pairs = SvCandidatePairs(
        first_rows=np.array([0, 1, 2, 2, 3], dtype=np.int64),
        second_rows=np.array([0, 0, 1, 2, 3], dtype=np.int64),
        size_ratios=np.ones(5),
        reciprocal_overlaps=np.ones(5),
        breakpoint_distances=np.full(5, -1, dtype=np.int64),
    )
    calibrations = [
        _calibration(pairing_z=12.0, accepted=True),
        _calibration(pairing_z=30.0, accepted=True),
        _calibration(pairing_z=8.0, accepted=True),
        _calibration(pairing_z=20.0, accepted=True),
        _calibration(pairing_z=40.0, accepted=False),
    ]

    chosen = resolve_one_to_one(pairs, calibrations)

    assert chosen.tolist() == [1, 3]


def test_degenerate_loci_carry_no_pairing_evidence() -> None:
    dosage = np.array([0.1, 0.9, 1.2, 0.0, 0.4, 1.8])
    constant_calls = np.ones(6)
    few_observed = np.array([True, True, True, False, False, False])
    posterior_variance = np.full(6, 0.1)

    constant = calibrate_two_sources(dosage, posterior_variance, constant_calls, np.ones(6, dtype=bool))
    too_few = calibrate_two_sources(dosage, posterior_variance, np.array([0, 1, 1, 0, 0, 2]), few_observed)
    # A second source that is an exact affine map of the first has a singular
    # covariance; the calibration stays finite.
    perfect = calibrate_two_sources(dosage, posterior_variance, 2 * dosage + 1, np.ones(6, dtype=bool))

    assert constant.pairing_z == 0.0 and not constant.accepted
    assert too_few.sample_count == 3 and too_few.pairing_z == 0.0 and not too_few.accepted
    assert np.isfinite(perfect.first_weight) and np.isfinite(perfect.second_weight)


def test_sequence_resolved_sites_start_after_the_shared_prefix() -> None:
    deleted = "ACGTTGCA" * 25
    inserted = "TTAGGC" * 20
    sites = sequence_resolved_sites(
        np.array(["chr1", "chr1", "chr1"]),
        np.array([1_000, 5_000, 9_000], dtype=np.int64),
        ["G" + deleted, "C", "T" + "A" * 120 + "G"],
        ["G", "C" + inserted, "T" + "C" * 90 + "G"],
    )

    assert sites.kinds.tolist() == ["DEL", "INS", "CPX"]
    assert sites.starts.tolist() == [1_001, 5_001, 9_001]
    assert sites.ends.tolist() == [1_201, 5_002, 9_121]
    assert sites.sizes.tolist() == [200, 120, 120]
    assert sites.duplications_are_insertions
    # The deletion lines up with the same event as GATK-SV writes it (POS 1000,
    # the padding base; END 1200, the last deleted base).
    gatksv = _sites([("chr1", 1_001, 1_201, 200, "DEL")], duplications_are_insertions=False)
    assert candidate_pairs(sites, gatksv).first_rows.tolist() == [0]

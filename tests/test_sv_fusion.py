"""Cross-source SV fusion: matching rules, two-source calibration, fused column."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.sv_fusion import (
    AnchorErrorModel,
    AnchorEstimate,
    SvCandidatePairs,
    SvSites,
    TwoSourceCalibration,
    berkson_reliability,
    calibrate_two_sources,
    candidate_pairs,
    fit_anchor_error_model,
    fused_dosage,
    mean_anchor,
    resolve_one_to_one,
    sequence_resolved_sites,
    shrunk_imputed_reliabilities,
    triad_log_reliability,
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

def _haplotypes(generator: np.random.Generator, sample_count: int, frequency: float) -> np.ndarray:
    return generator.random((sample_count, 2)) < frequency


def _draw_dosage(generator: np.random.Generator, haplotypes: np.ndarray, frequency: float, accuracy: float) -> np.ndarray:
    """A confident posterior draw: each haplotype is its true allele with probability ``accuracy``,
    otherwise an independent draw from the population frequency. r2_A = accuracy^2, rho_A = accuracy."""
    informed = generator.random(haplotypes.shape) < accuracy
    return np.where(informed, haplotypes, generator.random(haplotypes.shape) < frequency).sum(axis=1).astype(np.float64)


def _berkson_dosage(generator: np.random.Generator, haplotypes: np.ndarray, frequency: float, accuracy: float) -> np.ndarray:
    """A calibrated posterior mean: each haplotype is known with probability ``accuracy``, else it is
    the population frequency. r2_A = rho_A = accuracy."""
    informed = generator.random(haplotypes.shape) < accuracy
    return np.where(informed, haplotypes.astype(np.float64), frequency).sum(axis=1)


def _calls(generator: np.random.Generator, haplotypes: np.ndarray, miss_rate: float, false_rate: float) -> np.ndarray:
    return np.where(
        haplotypes, generator.random(haplotypes.shape) > miss_rate, generator.random(haplotypes.shape) < false_rate
    ).sum(axis=1).astype(np.float64)


def _squared_correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(first, second)[0, 1] ** 2)


def _oracle(genotype: np.ndarray, first: np.ndarray, second: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Least squares of the true genotype on the available sources, per no-call group."""
    fused = np.empty_like(genotype, dtype=np.float64)
    for rows, design in (
        (observed, np.column_stack([np.ones(observed.sum()), first[observed], second[observed]])),
        (~observed, np.column_stack([np.ones((~observed).sum()), first[~observed]])),
    ):
        coefficients, *_ = np.linalg.lstsq(design, genotype[rows].astype(np.float64), rcond=None)
        fused[rows] = design @ coefficients
    return fused


def _slope(response: np.ndarray, regressor: np.ndarray) -> float:
    centred = regressor - regressor.mean()
    return float(centred @ (response - response.mean()) / (centred @ centred))


@pytest.mark.parametrize(
    ("dosage", "accuracy"),
    [(_draw_dosage, 0.64), (_draw_dosage, 0.93), (_berkson_dosage, 0.6)],
    ids=["draw-vntr", "draw-outside-tr", "berkson"],
)
def test_given_its_reliability_the_fused_column_matches_the_oracle(dosage, accuracy) -> None:
    generator = np.random.default_rng(7)
    sample_count, frequency = 60_000, 0.2
    haplotypes = _haplotypes(generator, sample_count, frequency)
    genotype = haplotypes.sum(axis=1).astype(np.float64)
    first = dosage(generator, haplotypes, frequency, accuracy)
    second = _calls(generator, haplotypes, 0.2, 0.002)
    observed = generator.random(sample_count) >= 0.05

    calibration = calibrate_two_sources(first, second, observed, _squared_correlation(first, genotype))
    fused = fused_dosage(calibration, first, second, observed)

    assert calibration.accepted
    fused_truth = _squared_correlation(fused, genotype)
    assert fused_truth >= _squared_correlation(_oracle(genotype, first, second, observed), genotype) - 0.002
    assert abs(calibration.fused_reliability - fused_truth) < 0.01
    assert abs(calibration.second_reliability - _squared_correlation(second[observed], genotype[observed])) < 0.02
    assert 0.97 < _slope(genotype, fused) < 1.03
    # Where the other source is a no-call, the recalibrated dosage is calibrated too.
    assert 0.95 < _slope(genotype[~observed], fused[~observed]) < 1.05


def test_the_berkson_closure_misreads_a_draw_like_dosage() -> None:
    # The closure this replaced read r2_A = V_A / V_G, which is 1 for a draw:
    # the other source then gets no weight (VNTR gap to the oracle about -0.31).
    generator = np.random.default_rng(9)
    sample_count, frequency = 60_000, 0.25
    haplotypes = _haplotypes(generator, sample_count, frequency)
    genotype = haplotypes.sum(axis=1).astype(np.float64)
    first = _draw_dosage(generator, haplotypes, frequency, 0.64)
    second = _calls(generator, haplotypes, 0.15, 0.002)
    observed = generator.random(sample_count) >= 0.05
    oracle_truth = _squared_correlation(_oracle(genotype, first, second, observed), genotype)

    assert berkson_reliability(first) > 0.95
    as_berkson = calibrate_two_sources(first, second, observed, berkson_reliability(first))
    with_reliability = calibrate_two_sources(first, second, observed, _squared_correlation(first, genotype))

    assert oracle_truth - _squared_correlation(fused_dosage(as_berkson, first, second, observed), genotype) > 0.1
    assert oracle_truth - _squared_correlation(fused_dosage(with_reliability, first, second, observed), genotype) < 0.002


def test_genotype_dependent_no_calls_shift_the_group_mean() -> None:
    # GATK-SV-style: carriers are no-calls far more often than non-carriers.
    generator = np.random.default_rng(21)
    sample_count, frequency = 80_000, 0.2
    haplotypes = _haplotypes(generator, sample_count, frequency)
    genotype = haplotypes.sum(axis=1).astype(np.float64)
    first = _draw_dosage(generator, haplotypes, frequency, 0.8)
    second = _calls(generator, haplotypes, 0.1, 0.002)
    observed = generator.random(sample_count) >= np.where(genotype > 0, 0.27, 0.02)

    calibration = calibrate_two_sources(first, second, observed, _squared_correlation(first, genotype))
    fused = fused_dosage(calibration, first, second, observed)

    oracle_truth = _squared_correlation(_oracle(genotype, first, second, observed), genotype)
    assert oracle_truth - _squared_correlation(fused, genotype) < 0.015
    assert abs(fused[~observed].mean() - genotype[~observed].mean()) < 0.05


def test_unrelated_records_fail_the_calibration_check() -> None:
    generator = np.random.default_rng(5)
    sample_count, frequency = 50_000, 0.1
    first = _draw_dosage(generator, _haplotypes(generator, sample_count, frequency), frequency, 0.9)
    unrelated = _calls(generator, _haplotypes(generator, sample_count, frequency), 0.1, 0.001)
    observed = np.ones(sample_count, dtype=bool)

    calibration = calibrate_two_sources(first, unrelated, observed, 0.8)

    assert not calibration.accepted
    with pytest.raises(ValueError, match="keep them as separate columns"):
        fused_dosage(calibration, first, unrelated, observed)


def test_a_reliability_the_pair_contradicts_is_rejected() -> None:
    # Given r2_A far below what the pair's correlation needs, the other
    # source's implied reliability corr^2 / r2_A passes the slack.
    generator = np.random.default_rng(13)
    sample_count, frequency = 40_000, 0.2
    haplotypes = _haplotypes(generator, sample_count, frequency)
    first = _draw_dosage(generator, haplotypes, frequency, 0.9)
    second = _calls(generator, haplotypes, 0.05, 0.001)
    observed = np.ones(sample_count, dtype=bool)

    calibration = calibrate_two_sources(first, second, observed, 0.05)

    assert calibration.pairing_z >= 5
    assert calibration.second_reliability > 1.25
    assert not calibration.accepted


def _calibration(pairing_z: float, accepted: bool) -> TwoSourceCalibration:
    return TwoSourceCalibration(
        sample_count=1000,
        first_reliability=0.8 if accepted else 1.5,
        first_slope=0.9,
        genotype_variance=0.1,
        observed_first_mean=0.1,
        observed_second_mean=0.1,
        missing_first_mean=0.1,
        observed_genotype_mean=0.1,
        missing_genotype_mean=0.1,
        first_weight=0.5,
        second_weight=0.5,
        missing_slope=0.9,
        second_slope=0.9,
        second_reliability=0.7,
        fused_reliability=0.9,
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
    first = np.array([0.1, 0.9, 1.2, 0.0, 0.4, 1.8])
    few_observed = np.array([True, True, True, False, False, False])

    constant = calibrate_two_sources(first, np.ones(6), np.ones(6, dtype=bool), 0.5)
    too_few = calibrate_two_sources(first, np.array([0, 1, 1, 0, 0, 2]), few_observed, 0.5)
    monomorphic = calibrate_two_sources(np.zeros(6), np.array([0, 1, 1, 0, 0, 2]), np.ones(6, dtype=bool), 0.5)
    # A second source that is an exact affine map of the first has a singular
    # covariance; the calibration stays finite.
    perfect = calibrate_two_sources(first, 2 * first + 1, np.ones(6, dtype=bool), 0.9)

    assert constant.pairing_z == 0.0 and not constant.accepted
    assert too_few.sample_count == 3 and too_few.pairing_z == 0.0 and not too_few.accepted
    assert monomorphic.pairing_z == 0.0 and not monomorphic.accepted
    assert np.isfinite(perfect.first_weight) and np.isfinite(perfect.second_weight)


# ---------------------------------------------------------------------------
# The imputed source's reliability: mean anchor, error model, shrinkage
# ---------------------------------------------------------------------------


def test_the_mean_anchor_recovers_a_draw_like_reliability_and_its_variance() -> None:
    generator = np.random.default_rng(31)
    sample_count, frequency, accuracy = 20_000, 0.15, 0.7
    estimates = []
    for _ in range(60):
        haplotypes = _haplotypes(generator, sample_count, frequency)
        first = _draw_dosage(generator, haplotypes, frequency, accuracy)
        second = _calls(generator, haplotypes, 0.2, 0.0)
        estimates.append(mean_anchor(first, second, np.ones(sample_count, dtype=bool), 0.0, 0.0))
    values = np.array([estimate.log_reliability for estimate in estimates])

    assert abs(np.exp(values.mean()) - accuracy**2) < 0.02
    # The delta-method variance matches the spread across replicates.
    assert 0.6 < np.mean([estimate.variance for estimate in estimates]) / values.var() < 1.6


def test_the_mean_anchor_removes_a_known_false_positive_intercept() -> None:
    generator = np.random.default_rng(33)
    sample_count, frequency, accuracy, false_rate = 50_000, 0.05, 0.7, 0.004
    haplotypes = _haplotypes(generator, sample_count, frequency)
    first = _draw_dosage(generator, haplotypes, frequency, accuracy)
    second = _calls(generator, haplotypes, 0.2, false_rate)
    observed = np.ones(sample_count, dtype=bool)

    ignored = mean_anchor(first, second, observed, 0.0, 0.0)
    corrected = mean_anchor(first, second, observed, false_rate, 1e-6)

    # A draw-like dosage right with probability a per haplotype has population r2_A = a^2.
    target = np.log(accuracy**2)
    assert abs(corrected.log_reliability - target) < abs(ignored.log_reliability - target)
    # Within four of the anchor's own delta-method standard errors of the population value.
    assert abs(corrected.log_reliability - target) < 4 * np.sqrt(corrected.variance)
    assert corrected.variance > mean_anchor(first, second, observed, false_rate, 0.0).variance


def test_the_mean_anchor_is_exact_on_a_population_with_false_positives() -> None:
    # Exact population moments: g ~ HWE(1/4); a read equal to g with probability 3/4, else each
    # other genotype with 1/8; A = E[g | read] (mean-calibrated); B calls a carrier haplotype with
    # sensitivity 4/5 and a non-carrier one with false-positive rate f = 1/10, independently of A
    # given g, so E[B | g] = 2 f + (s - f) g. Every cell probability is a multiple of 1/12800, so
    # the rows below are the population itself and the anchor must recover r2_A exactly.
    sensitivity, false_rate = 0.8, 0.1
    prior = np.array([9, 6, 1]) / 16
    reads = np.full((3, 3), 1 / 8) + np.eye(3) * (3 / 4 - 1 / 8)
    posterior_mean = (prior[:, None] * reads * np.arange(3)[:, None]).sum(axis=0) / (prior[:, None] * reads).sum(axis=0)
    carrier, non_carrier = np.array([1 - sensitivity, sensitivity]), np.array([1 - false_rate, false_rate])
    genotypes, dosages, calls, weights = [], [], [], []
    for genotype in range(3):
        haplotypes = [carrier] * genotype + [non_carrier] * (2 - genotype)
        call_distribution = np.convolve(haplotypes[0], haplotypes[1])
        for read in range(3):
            for call in range(3):
                genotypes.append(genotype)
                dosages.append(posterior_mean[read])
                calls.append(call)
                weights.append(prior[genotype] * reads[genotype, read] * call_distribution[call])
    counts = np.rint(np.array(weights) * 12_800).astype(np.int64)
    # Each weight is a product of four rounded factors, so it sits within 4 eps of its multiple.
    np.testing.assert_allclose(counts, np.array(weights) * 12_800, rtol=4 * np.finfo(float).eps)
    genotype = np.repeat(np.array(genotypes, dtype=np.float64), counts)
    first = np.repeat(np.array(dosages), counts)
    second = np.repeat(np.array(calls, dtype=np.float64), counts)

    anchor = mean_anchor(first, second, np.ones(first.shape[0], dtype=bool), false_rate, 0.0)

    # Algebraically exact: only floating-point summation over the rows separates the two.
    np.testing.assert_allclose(np.exp(anchor.log_reliability), _squared_correlation(first, genotype), rtol=first.shape[0] * np.finfo(float).eps)


def test_an_undefined_anchor_has_infinite_variance() -> None:
    first = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 0.0])
    anticorrelated = 2.0 - first
    anchor = mean_anchor(first, anticorrelated, np.ones(6, dtype=bool), 0.0, 0.0)

    assert np.isnan(anchor.log_reliability) and anchor.variance == np.inf


def test_the_triad_corrects_for_noisy_truth() -> None:
    generator = np.random.default_rng(41)
    sample_count, frequency = 3_000, 0.3
    haplotypes = _haplotypes(generator, sample_count, frequency)
    genotype = haplotypes.sum(axis=1).astype(np.float64)
    first = _draw_dosage(generator, haplotypes, frequency, 0.7)
    truth_one = _calls(generator, haplotypes, 0.15, 0.02)
    truth_two = _calls(generator, haplotypes, 0.15, 0.02)

    estimate, variance = triad_log_reliability(first, truth_one, truth_two)

    naive = np.log(_squared_correlation(first, truth_one))
    true_value = np.log(_squared_correlation(first, genotype))
    assert abs(estimate - true_value) < abs(naive - true_value)
    assert abs(estimate - true_value) < 3 * np.sqrt(variance) + 0.02
    assert 0.0 < variance < 0.05


def test_the_error_model_recovers_the_anchor_bias_and_excess_spread() -> None:
    generator = np.random.default_rng(43)
    loci = 4_000
    truth = np.log(generator.uniform(0.3, 0.9, loci))
    sampling = np.full(loci, 0.01)
    truth_sampling = np.full(loci, 0.02)
    anchors = [
        AnchorEstimate(log_reliability=float(value), variance=0.01)
        for value in truth + 0.2 + generator.normal(0.0, np.sqrt(0.05 + 0.01), loci)
    ]
    noisy_truth = truth + generator.normal(0.0, np.sqrt(0.02), loci)

    model = fit_anchor_error_model(anchors, noisy_truth, truth_sampling, berkson=False)

    assert abs(model.bias - 0.2) < 0.02
    assert abs(model.excess_variance - 0.05) < 0.01
    assert not model.berkson
    with pytest.raises(ValueError, match="at least two truth loci"):
        fit_anchor_error_model(anchors[:1], noisy_truth[:1], sampling[:1], berkson=False)


def test_shrinkage_beats_both_the_anchor_and_the_prediction() -> None:
    generator = np.random.default_rng(47)
    loci = 3_000
    prior = np.log(np.full(loci, 0.5))
    truth = prior + generator.normal(0.0, 0.3, loci)
    model = AnchorErrorModel(bias=0.1, excess_variance=0.04, berkson=False)
    anchors = [
        AnchorEstimate(log_reliability=float(value), variance=0.01)
        for value in truth + model.bias + generator.normal(0.0, np.sqrt(0.04 + 0.01), loci)
    ]

    shrunk = np.log(shrunk_imputed_reliabilities(anchors, prior, model))

    anchor_values = np.array([anchor.log_reliability for anchor in anchors]) - model.bias
    error = np.mean((shrunk - truth) ** 2)
    assert error < np.mean((anchor_values - truth) ** 2)
    assert error < np.mean((prior - truth) ** 2)


def test_shrinkage_keeps_the_prediction_where_the_anchor_says_nothing() -> None:
    prior = np.log(np.array([0.4, 0.6, 0.8]))
    undefined = AnchorEstimate(log_reliability=float("nan"), variance=float("inf"))
    model = AnchorErrorModel(bias=0.0, excess_variance=0.0, berkson=False)

    lone = shrunk_imputed_reliabilities([undefined, AnchorEstimate(-0.1, 0.01), undefined], prior, model)
    swamped = shrunk_imputed_reliabilities(
        [AnchorEstimate(-2.0, 0.01), AnchorEstimate(-0.1, 0.01), AnchorEstimate(-1.0, 0.01)],
        prior,
        AnchorErrorModel(bias=0.0, excess_variance=100.0, berkson=False),
    )

    np.testing.assert_allclose(lone, np.exp(prior))
    np.testing.assert_allclose(swamped, np.exp(prior))


def test_a_berkson_dosage_carries_its_own_reliability() -> None:
    generator = np.random.default_rng(51)
    sample_count, frequency = 60_000, 0.2
    haplotypes = _haplotypes(generator, sample_count, frequency)
    first = _berkson_dosage(generator, haplotypes, frequency, 0.55)

    assert abs(berkson_reliability(first) - _squared_correlation(first, haplotypes.sum(axis=1))) < 0.02
    assert np.isnan(berkson_reliability(np.zeros(10)))


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

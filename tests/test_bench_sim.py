"""bench-sim: the benchmark's data-free building blocks.

Statistical checks use fixed seeds; their bounds are Monte Carlo standard errors times the normal quantile of
FALSE_FAILURE_PROBABILITY, so a correct implementation fails with at most that probability over seeds.
"""

from __future__ import annotations

import gzip

import numpy as np
from scipy.stats import norm

from benchmarks.bench_sim import cohort, harness, measurement, measurement_beagle, truth
from benchmarks.bench_sim.annotations import merged_intervals, overlaps
from sv_pgs.dosage_store import encode_dosage_milli

FALSE_FAILURE_PROBABILITY = 1e-9
Z_BOUND = float(norm.isf(FALSE_FAILURE_PROBABILITY / 2))
EPSILON = float(np.finfo(np.float64).eps)
# Sharma et al. 2025 (Nat Commun, doi:10.1038/s41467-025-59351-8): mean continental ancestry in All of Us,
# West Asian folded into EUR and Oceanian dropped, renormalized. Order: AFR, AMR, EAS, EUR, SAS.
PUBLISHED_MEANS = np.array([19.51, 6.33, 2.57, 66.37 + 1.95, 3.05])
PUBLISHED_MEANS = PUBLISHED_MEANS / PUBLISHED_MEANS.sum()


def test_dosage_codes_match_the_store_encoder() -> None:
    milli = np.arange(2001)
    assert np.array_equal(measurement.encode_milli(milli), encode_dosage_milli(milli))


def test_group_weights_reproduce_the_published_ancestry_means() -> None:
    weights = np.array([spec[0] for spec in cohort.GROUPS.values()])
    means = np.array([spec[1] for spec in cohort.GROUPS.values()])
    implied = weights @ means / weights.sum()
    # The weights are published to 4 decimals: each rounding moves a mean by at most 0.5e-4 per group.
    assert np.all(np.abs(implied - PUBLISHED_MEANS) <= 0.5e-4 * len(cohort.GROUPS) + EPSILON)


def test_cohort_proportions_average_to_the_group_means() -> None:
    rng = np.random.default_rng(1)
    size = 200_000
    group, proportions, generations = cohort.draw_cohort(size, rng)
    for index, (name, (_, mean, admixture)) in enumerate(cohort.GROUPS.items()):
        members = proportions[group == index]
        assert np.allclose(members.sum(axis=1), 1.0, atol=len(cohort.SUPERPOPS) * EPSILON)
        standard_error = members.std(axis=0) / np.sqrt(members.shape[0])
        assert np.all(np.abs(members.mean(axis=0) - np.asarray(mean)) <= Z_BOUND * standard_error + EPSILON), name
        assert np.all(generations[group == index] == admixture)


def test_haplotype_paths_copy_only_from_the_tract_ancestry() -> None:
    rng = np.random.default_rng(2)
    cm = np.linspace(0.0, 70.0, 20_000)
    donor_columns = {index: np.arange(100 * index, 100 * index + 100) for index in range(len(cohort.SUPERPOPS))}
    for proportions, generations in (((1.0, 0, 0, 0, 0), 0.0), ((0.5, 0, 0, 0.5, 0), 7.0)):
        starts, donors, ancestries = cohort.haplotype_path(cm, np.asarray(proportions), generations, donor_columns, rng)
        assert starts[0] == 0 and np.all(np.diff(starts) >= 0)
        assert set(np.unique(ancestries)) <= set(np.flatnonzero(np.asarray(proportions) > 0))
        for donor, ancestry in zip(donors, ancestries):
            assert donor in donor_columns[int(ancestry)]


def test_interval_overlap_handles_nested_intervals() -> None:
    starts = np.array([10, 12, 40])
    ends = np.array([30, 14, 50])
    query_start = np.array([0, 15, 29, 30, 45, 55])
    query_end = np.array([10, 16, 31, 40, 46, 60])
    expected = np.array([False, True, True, False, True, False])
    assert np.array_equal(cohort.overlaps_interval(starts, ends, query_start, query_end), expected)
    assert np.array_equal(overlaps(starts, ends, query_start, query_end), expected)


def test_repeat_intervals_merge_overlaps(tmp_path) -> None:
    path = tmp_path / "repeats.txt.gz"
    with gzip.open(path, "wt") as handle:
        for start, end in ((100, 200), (150, 250), (300, 310), (10, 20)):
            handle.write(f"0\tchr22\t{start}\t{end}\tx\n")
        handle.write("0\tchr21\t5\t9\tx\n")
    merged_start, merged_end = merged_intervals(path, "chr22")
    assert merged_start.tolist() == [10, 100, 300]
    assert merged_end.tolist() == [20, 250, 310]


def test_hom_ref_calls_follow_the_gvcf_block_rule() -> None:
    rng = np.random.default_rng(3)
    genotype = np.zeros(200_000, dtype=np.intp)
    pl = measurement.simulated_pl(genotype, measurement.BASE_ERROR[0], rng)
    called_hom_ref = pl[:, 0] == 0
    assert called_hom_ref.any()
    block = pl[called_hom_ref]
    assert block[:, 1].max() <= measurement.GQ_CAP // 5
    assert block[:, 2].max() <= measurement.HOMREF_HOM_ALT_CAP
    assert np.array_equal(block[:, 2], np.minimum(2 * block[:, 1], measurement.HOMREF_HOM_ALT_CAP))
    assert np.all(pl.min(axis=1) == 0)


def test_variant_calls_keep_their_likelihoods() -> None:
    rng = np.random.default_rng(4)
    genotype = np.ones(50_000, dtype=np.intp)
    pl = measurement.simulated_pl(genotype, measurement.BASE_ERROR[0], rng)
    assert (pl[:, 1] == 0).any()
    assert np.all(pl <= measurement.PL_CAP)


def test_pl_text_round_trips() -> None:
    rng = np.random.default_rng(5)
    pl = rng.integers(0, measurement.PL_CAP + 1, size=(1_000, 3))
    text = measurement.pl_text(pl).tobytes().decode()
    fields = text.rstrip("\t").split("\t")
    parsed = np.array([[int(value) for value in field.split(",")] for field in fields])
    assert np.array_equal(parsed, pl)


def test_shape_families_have_unit_variance() -> None:
    rng = np.random.default_rng(6)
    count = 400_000
    cases = {
        "gaussian": {}, "laplace": {}, "two_point": {}, "sparse_mixture": {},
        "variance_grid": {"grid_weights": [0.5, 0.3, 0.2]}, "lognormal_scale": {"tau": 0.5},
        "student_t": {"nu": 8.0},
    }
    for shape, params in cases.items():
        draws = truth.unit_variance_draws(rng, shape, count, params)
        centred = draws - draws.mean()
        variance = float(np.mean(centred**2))
        fourth = float(np.mean(centred**4))
        standard_error = np.sqrt(max(fourth - variance**2, 0.0) / count)
        assert abs(variance - 1.0) <= Z_BOUND * standard_error + EPSILON, shape


def test_scenario_parameters_are_deterministic_and_in_range() -> None:
    for seed in range(200):
        params = truth.draw_parameters(seed)
        assert params == truth.draw_parameters(seed)
        assert 0.01 <= params["h2"] <= 0.2
        assert 1e-4 <= params["pi"] <= 3e-2
        assert -1.0 <= params["alpha_snv"] <= 0.0
        assert abs(params["alpha_sv"] - params["alpha_snv"]) <= 0.5
        assert params["shape"] in truth.SHAPES and params["sv_shape"] in truth.SHAPES
        assert params["sv_mode"] in truth.MODES and params["annotation_mode"] in truth.MODES
        assert 0.01 <= params["prevalence"] <= 0.3


def test_sealed_seeds_are_deterministic_and_distinct() -> None:
    master = "ab" * 32
    seeds = truth.sealed_seeds(master, 64)
    assert seeds == truth.sealed_seeds(master, 64)
    assert len(set(seeds)) == 64
    assert truth.sealed_seeds("cd" * 32, 64) != seeds


def test_random_spline_is_centred() -> None:
    rng = np.random.default_rng(7)
    values = rng.gamma(2.0, 3.0, size=50_000)
    curve = truth.random_spline(rng, values, 0.4)
    assert abs(curve.mean()) <= values.size * EPSILON * np.abs(curve).max()
    assert curve.std() > 0


def test_incremental_r2_of_an_exact_prediction() -> None:
    rng = np.random.default_rng(8)
    covariates = rng.standard_normal((5_000, 3))
    covariate_part = covariates @ np.array([0.5, -0.2, 0.1])
    prediction = rng.standard_normal(5_000)
    design = np.column_stack([np.ones(5_000), covariates])
    prediction -= design @ np.linalg.lstsq(design, prediction, rcond=None)[0]
    outcome = covariate_part + prediction
    gain, slope = harness.incremental_r2(outcome, prediction, covariates)
    base = covariate_part - covariate_part.mean()
    total = outcome - outcome.mean()
    expected = 1.0 - (base @ base) / (total @ total)
    # Backward-stable least squares: error of order cond(design) * eps, accumulated over n terms.
    tolerance = np.linalg.cond(np.column_stack([design, prediction])) * EPSILON * outcome.size
    assert abs(gain - expected) <= tolerance
    assert abs(slope - 1.0) <= tolerance


def test_auc_matches_pairwise_counting() -> None:
    rng = np.random.default_rng(9)
    outcome = (rng.random(400) < 0.3).astype(float)
    score = rng.standard_normal(400) + outcome
    positives, negatives = score[outcome > 0.5], score[outcome < 0.5]
    expected = float((positives[:, None] > negatives[None, :]).mean())
    assert abs(harness.auc(outcome, score) - expected) <= positives.size * negatives.size * EPSILON


def test_beagle_alleles_are_unique_per_record() -> None:
    assert measurement_beagle.beagle_allele("<INS>", 7) == "<INS:v7>"
    assert measurement_beagle.beagle_allele("<INS>", 7) != measurement_beagle.beagle_allele("<INS>", 8)
    assert measurement_beagle.beagle_allele("ACGT", 7) == "ACGT"

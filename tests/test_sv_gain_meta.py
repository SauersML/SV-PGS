"""benchmarks/bench_real/sv_gain_meta.py: the combined SV-gain test is calibrated where no trait gains and finds a
planted gain, on small synthetic traits sharing people and families."""

from __future__ import annotations

import numpy as np
from scipy import stats

from benchmarks.bench_real import sv_gain_meta
from benchmarks.bench_real.sv_gain_meta import AblationTrait


def _synthetic(generator: np.random.Generator, gain: float, people: int = 450, traits: int = 3) -> tuple[list[AblationTrait], np.ndarray, np.ndarray]:
    """Traits on overlapping held-out people in families of 1 to 3. Each baseline is g + e1 and each extra part
    e2 - e1 + gain * h, so the candidate is g + e2 + gain * h: with gain 0 the two scores are exchangeable and the
    population gain is 0; h is part of the outcome the baseline misses."""
    sizes = generator.integers(1, 4, size=people)
    families = np.repeat(np.arange(people), sizes)[:people]
    family_count = int(families.max()) + 1
    ancestry = np.array(["eur", "afr", "amr"], dtype=object)[generator.integers(0, 3, size=family_count)][families].astype(str)
    components = generator.normal(size=(people, 2))
    shared = generator.normal(size=people)
    family_effect = generator.normal(size=family_count)[families] * 0.5
    result = []
    for index in range(traits):
        own = np.sort(generator.choice(people, size=int(people * 0.85), replace=False))
        g = 0.6 * shared[own] + 0.8 * generator.normal(size=own.shape[0])
        h = generator.normal(size=own.shape[0])
        target = g + 0.7 * h + 0.3 * components[own, 0] + family_effect[own] + 2.0 * generator.normal(size=own.shape[0])
        first, second = generator.normal(size=(2, own.shape[0]))
        covariates = np.column_stack([np.ones(own.shape[0]), components[own]])
        result.append(AblationTrait(f"trait{index}", False, own, target, covariates, g + first, second - first + gain * h))
    return result, families, ancestry


def test_union_clusters_join_people_one_run_puts_in_a_family() -> None:
    # Run A: people 0-1 one family, 2 alone; run B: people 1-3 one family; 4 is in neither run's family with others.
    clusters = sv_gain_meta.union_clusters([(np.array([0, 1, 2]), np.array([7, 7, 8])), (np.array([1, 3, 4]), np.array([5, 5, 6]))], 5)
    assert clusters[0] == clusters[1] == clusters[3]
    assert len({clusters[0], clusters[2], clusters[4]}) == 3


def test_the_gain_at_unit_weights_is_the_difference_of_partial_r2() -> None:
    traits, families, _ancestry = _synthetic(np.random.default_rng(1), 0.5, traits=1)
    trait = traits[0]
    gain, base = sv_gain_meta.gains(np.ones((1, families.shape[0])), trait)

    def partial(score: np.ndarray) -> float:
        projection = trait.covariates @ np.linalg.lstsq(trait.covariates, np.column_stack([trait.target, score]), rcond=None)[0]
        residual = np.column_stack([trait.target, score]) - projection
        return float(np.corrcoef(residual, rowvar=False)[0, 1] ** 2)

    expected = partial(trait.baseline + trait.extra) - partial(trait.baseline)
    assert np.isclose(gain[0], expected, rtol=1e-9, atol=1e-12)
    assert np.isclose(base[0], partial(trait.baseline), rtol=1e-9)


def test_stouffer_and_acat_hold_their_level_for_correlated_traits() -> None:
    generator = np.random.default_rng(2)
    correlation = np.array([[1.0, 0.5, 0.2, 0.1], [0.5, 1.0, 0.3, 0.1], [0.2, 0.3, 1.0, 0.4], [0.1, 0.1, 0.4, 1.0]])
    z = generator.multivariate_normal(np.zeros(4), correlation, size=4000)
    p = 2.0 * stats.norm.sf(np.abs(z))
    stouffer = np.array([sv_gain_meta.stouffer(row, correlation)[1] for row in z])
    acat = np.array([sv_gain_meta.acat(row) for row in p])
    omnibus = np.array([sv_gain_meta.acat([a, b]) for a, b in zip(stouffer, acat)])
    assert 0.04 <= np.mean(stouffer < 0.05) <= 0.06
    assert 0.035 <= np.mean(acat < 0.05) <= 0.065
    assert np.mean(omnibus < 0.05) <= 0.075
    # the Stouffer p is uniform, not only at 0.05
    assert stats.kstest(stouffer, "uniform").pvalue > 0.01


def test_no_gain_is_found_where_none_was_planted_and_a_planted_gain_is_found() -> None:
    generator = np.random.default_rng(3)
    rejections, statistics = [], []
    for _ in range(80):
        traits, families, ancestry = _synthetic(generator, 0.0)
        result = sv_gain_meta.analyse(traits, families, ancestry, int(generator.integers(1 << 30)), resamples=100, permutations=0)
        rejections.append(result["p_stouffer"] < 0.05)
        statistics.append(result["Z"])
    assert np.mean(rejections) <= 0.15
    assert abs(np.mean(statistics)) < 0.4 and 0.7 < np.std(statistics) < 1.4
    traits, families, ancestry = _synthetic(np.random.default_rng(4), 0.6)
    planted = sv_gain_meta.analyse(traits, families, ancestry, 5, resamples=100, permutations=10, permutation_resamples=40)
    assert planted["p_stouffer_one_sided"] < 0.01 and np.all(planted["estimate"] > 0)
    calibration = planted["calibration"]
    # the permuted extra part keeps no link to the outcome, so the observed Z lies above every permuted one
    assert calibration["permutations"] == 10 and calibration["p_permutation_Z_upper"] == 1 / 11
    assert np.all(np.isfinite(calibration["z_sd"]))
    lines = sv_gain_meta.report(planted, traits, ancestry)
    assert any(line.startswith("PRIMARY Stouffer") for line in lines)


def test_a_permutation_moves_whole_families_within_their_stratum() -> None:
    traits, families, ancestry = _synthetic(np.random.default_rng(6), 0.0, traits=1)
    trait = traits[0]
    blocks = sv_gain_meta.family_blocks(trait, families, ancestry)
    values = np.arange(trait.people.shape[0], dtype=np.float64)
    moved = sv_gain_meta.permuted(values, blocks, np.random.default_rng(7).random(int(families.max()) + 1))
    for ids, members in zip(blocks.clusters, blocks.members):
        # every family of the stratum receives another family's whole block, in its member order
        assert sorted(map(tuple, values[members])) == sorted(map(tuple, moved[members]))
    assert sorted(moved) == sorted(values) and not np.array_equal(moved, values)


def test_reml_finds_no_spread_in_equal_estimates_and_some_in_spread_ones() -> None:
    even = sv_gain_meta.reml_hksj(np.array([1.0, 1.1, 0.9, 1.0]), np.array([0.04, 0.05, 0.06, 0.05]))
    assert even["tau_squared"] == 0.0 and even["ci_low"] < 1.0 < even["ci_high"]
    weights = 1.0 / np.array([0.04, 0.05, 0.06, 0.05])
    assert np.isclose(even["estimate"], np.sum(weights * np.array([1.0, 1.1, 0.9, 1.0])) / weights.sum())
    spread = sv_gain_meta.reml_hksj(np.array([0.0, 3.0, -2.0, 5.0]), np.array([0.04, 0.05, 0.06, 0.05]))
    assert spread["tau_squared"] > 1.0
    # the HKSJ interval is a t interval on k - 1 = 3 degrees of freedom
    half = spread["ci_high"] - spread["estimate"]
    assert np.isclose(half, stats.t.ppf(0.975, 3) * spread["standard_error"])

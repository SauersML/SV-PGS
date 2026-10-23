"""benchmarks/bench_real/sv_gain_meta.py: the combined with_sv vs snv_indel test is calibrated where no trait gains,
finds a planted gain and bounds it, on small synthetic traits sharing people and families."""

from __future__ import annotations

import numpy as np
from scipy import stats

from benchmarks.bench_real import sv_gain_meta
from benchmarks.bench_real.sv_gain_meta import PairedTrait


def _synthetic(generator: np.random.Generator, gain: float, people: int = 450, traits: int = 3) -> tuple[list[PairedTrait], np.ndarray, np.ndarray]:
    """Traits on overlapping held-out people in families of 1 to 3. The snv_indel arm scores g + e1 and the with_sv arm
    g + e2 + gain * h, h a part of the outcome the snv_indel arm misses: with gain 0 the arms are exchangeable."""
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
        result.append(PairedTrait(f"trait{index}", False, own, target, covariates, g + first, g + second + gain * h))
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

    assert np.isclose(gain[0], partial(trait.candidate) - partial(trait.baseline), rtol=1e-9, atol=1e-12)
    assert np.isclose(base[0], partial(trait.baseline), rtol=1e-9)
    # every person swapped turns the gain around exactly
    swapped, _ = sv_gain_meta.gains(np.ones((1, families.shape[0])), trait, np.ones(trait.people.shape[0], dtype=bool))
    assert np.isclose(swapped[0], -gain[0], rtol=1e-9)


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


def test_no_gain_is_found_where_none_was_planted_and_a_planted_gain_is_found_and_bounded() -> None:
    generator = np.random.default_rng(3)
    rejections, statistics, covered = [], [], []
    for _ in range(80):
        traits, families, ancestry = _synthetic(generator, 0.0)
        result = sv_gain_meta.analyse(traits, families, ancestry, int(generator.integers(1 << 30)), resamples=100, permutations=0)
        rejections.append(result["p_stouffer"] < 0.05)
        statistics.append(result["Z"])
        # the true gain is 0, so each one-sided upper bound lies above it 95% of the time
        covered.extend(result["upper_95"] >= 0.0)
    assert np.mean(rejections) <= 0.15
    assert abs(np.mean(statistics)) < 0.4 and 0.7 < np.std(statistics) < 1.4
    assert np.mean(covered) >= 0.88
    traits, families, ancestry = _synthetic(np.random.default_rng(4), 0.6)
    planted = sv_gain_meta.analyse(traits, families, ancestry, 5, resamples=100, permutations=20, permutation_resamples=40)
    assert planted["p_stouffer_one_sided"] < 0.01 and np.all(planted["estimate"] > 0)
    assert planted["common"]["p_one_sided"] < 0.01 and planted["common"]["upper_95"] > planted["common"]["estimate"] > 0
    assert calibration_ok(planted["calibration"])
    assert np.all(planted["upper_95"] > planted["estimate"]) and np.all(planted["relative_upper_95"] > planted["relative"])
    calibration = planted["calibration"]
    # swapped arms are exchangeable, so the observed Z lies above every swapped one
    assert calibration["permutations"] == 20 and calibration["p_permutation_Z_upper"] == 1 / 21
    assert abs(calibration["Z_mean"]) < 1.0 and np.all(np.isfinite(calibration["z_sd"]))
    lines = sv_gain_meta.report(planted, traits)
    assert any(line.startswith("PRIMARY common relative gain") for line in lines) and any("at most" in line for line in lines)
    assert any(line.startswith("original primary, now secondary: Stouffer") for line in lines)
    assert sv_gain_meta.report(planted, traits, "ablation")[0] == sv_gain_meta.ABLATION_LABEL and lines[0] != sv_gain_meta.ABLATION_LABEL


def calibration_ok(calibration: dict[str, object]) -> bool:
    # swapped arms are exchangeable, so the observed primary z lies above every swapped one
    return calibration["p_permutation_common_upper"] == 1 / 21 and abs(calibration["common_z_mean"]) < 1.0


def test_the_common_relative_gain_holds_its_level_and_its_interval_covers_0_without_a_gain() -> None:
    # 1,500 people per trait: at 450 every statistic here, the Stouffer Z too, spreads about 15% wider than its bootstrap
    # SE says, while the All of Us traits hold 4,000 to 8,000 held-out people each
    generator = np.random.default_rng(11)
    statistics, covered = [], []
    for _ in range(50):
        traits, families, ancestry = _synthetic(generator, 0.0, people=1500)
        result = sv_gain_meta.analyse(traits, families, ancestry, int(generator.integers(1 << 30)), resamples=100, permutations=0)
        statistics.append(result["common"]["z"])
        covered.append(result["common"]["upper_95"] >= 0.0)
    assert abs(np.mean(statistics)) < 0.4 and 0.7 < np.std(statistics) < 1.4
    assert np.mean(np.array(statistics) > stats.norm.isf(0.05)) <= 0.12 and np.mean(covered) >= 0.88


def test_the_common_relative_gain_weights_traits_by_r2_over_the_gain_variance() -> None:
    estimates, bases = np.array([0.002, -0.001]), np.array([0.05, 0.005])
    draws = np.random.default_rng(12).normal(estimates, [0.001, 0.0005], size=(4000, 2))
    found = sv_gain_meta.common_relative(estimates, bases, draws, np.tile(bases, (4000, 1)))
    levels = bases + estimates / 2.0
    weights = levels / np.var(draws, axis=0, ddof=1)
    assert np.isclose(found["estimate"], weights @ estimates / (weights @ levels))
    assert np.isclose(found["weight_share"].sum(), 1.0) and found["weight_share"][0] > found["weight_share"][1]
    # a trait whose arms are the same score (no variance) gets no weight
    flat = sv_gain_meta.common_relative(estimates, bases, np.column_stack([draws[:, 0], np.full(4000, -0.001)]), np.tile(bases, (4000, 1)))
    assert flat["weight_share"][1] == 0.0 and np.isclose(flat["estimate"], estimates[0] / levels[0])


def _run(name: str, ids: list[str], target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    count = len(ids)
    return {"name": name, "binary": False, "seed": 1, "ids": np.array(ids, dtype=object), "families": np.arange(count),
            "ancestry": np.array(["eur"] * count, dtype=object), "target": target, "covariates": np.ones((count, 1)),
            "baseline": baseline, "candidate": candidate}


def test_pairs_of_one_trait_are_averaged_per_arm_on_their_common_people() -> None:
    generator = np.random.default_rng(13)
    target = generator.normal(size=6)
    first = _run("ldl", ["a", "b", "c", "d", "e", "f"], target, generator.normal(size=6), generator.normal(size=6))
    second = _run("ldl", ["f", "e", "d", "c", "b", "x"], np.r_[target[::-1][:5], 9.0], 3.0 * generator.normal(size=6), generator.normal(size=6) + 5.0)
    merged = sv_gain_meta.merge_pairs([first, second])
    assert merged["ids"].tolist() == ["b", "c", "d", "e", "f"] and merged["pairs"] == 2
    rows_second = [4, 3, 2, 1, 0]

    def standardized(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / values.std()

    for arm in ("baseline", "candidate"):
        expected = (standardized(first[arm][1:]) + standardized(second[arm][rows_second])) / 2.0
        assert np.allclose(merged[arm], expected)
    assert np.allclose(merged["target"], target[1:])
    broken = _run("ldl", ["b", "c"], np.array([0.0, 0.0]), np.zeros(2), np.ones(2))
    try:
        sv_gain_meta.merge_pairs([first, broken])
    except ValueError:
        pass
    else:
        raise AssertionError("runs whose outcomes disagree were merged")


def test_excluded_people_leave_every_per_person_array() -> None:
    run = _run("gout", ["a", "b", "c"], np.array([1.0, 2.0, 3.0]), np.zeros(3), np.ones(3))
    kept, lost = sv_gain_meta.excluded(run, {"b", "z"})
    assert lost == 1 and kept["ids"].tolist() == ["a", "c"] and kept["target"].tolist() == [1.0, 3.0] and kept["covariates"].shape == (2, 1)
    only, dropped = sv_gain_meta.excluded(run, {"b", "z"}, keep=True)
    assert dropped == 2 and only["ids"].tolist() == ["b"] and only["baseline"].tolist() == [0.0]


def test_one_family_coin_swaps_all_its_members_in_every_trait() -> None:
    traits, families, _ancestry = _synthetic(np.random.default_rng(6), 0.0)
    coins = np.random.default_rng(7).random(int(families.max()) + 1) < 0.5
    for trait in traits:
        swap = sv_gain_meta.family_swaps(trait, families, coins)
        own = families[trait.people]
        for family in np.unique(own):
            assert len(set(swap[own == family])) == 1 and swap[own == family][0] == coins[family]


def test_reml_finds_no_spread_in_equal_estimates_and_some_in_spread_ones() -> None:
    even = sv_gain_meta.reml_hksj(np.array([1.0, 1.1, 0.9, 1.0]), np.array([0.04, 0.05, 0.06, 0.05]))
    assert even["tau_squared"] == 0.0 and even["ci_low"] < 1.0 < even["upper_95"] < even["ci_high"]
    weights = 1.0 / np.array([0.04, 0.05, 0.06, 0.05])
    assert np.isclose(even["estimate"], np.sum(weights * np.array([1.0, 1.1, 0.9, 1.0])) / weights.sum())
    spread = sv_gain_meta.reml_hksj(np.array([0.0, 3.0, -2.0, 5.0]), np.array([0.04, 0.05, 0.06, 0.05]))
    assert spread["tau_squared"] > 1.0
    # the HKSJ interval is a t interval on k - 1 = 3 degrees of freedom
    assert np.isclose(spread["ci_high"] - spread["estimate"], stats.t.ppf(0.975, 3) * spread["standard_error"])
    assert np.isclose(spread["upper_95"] - spread["estimate"], stats.t.ppf(0.95, 3) * spread["standard_error"])
    single = sv_gain_meta.reml_hksj(np.array([0.5]), np.array([0.01]))
    assert single["estimate"] == 0.5 and np.isnan(single["upper_95"])


def test_one_trait_alone_gives_its_own_test() -> None:
    traits, families, ancestry = _synthetic(np.random.default_rng(8), 0.3, traits=1)
    result = sv_gain_meta.analyse(traits, families, ancestry, 9, resamples=100, permutations=0)
    assert np.isclose(result["Z"], result["z"][0]) and np.isclose(result["p_acat"], result["p"][0], rtol=1e-6)
    assert np.isfinite(result["absolute_pooled"]["fixed"].estimate)


def test_a_null_arm_or_an_arm_fitted_under_another_key_or_seed_excludes_a_pair_and_uncertified_is_flagged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sv_gain_meta, "OPS", str(tmp_path))
    for name, key, seed in (("aou-p1", "k", 1), ("aou-p2", "k", 1), ("aou-q1", "k", 1), ("aou-q2", "other", 2)):
        (tmp_path / name / "run" / ("fit" if name.endswith("1") else "fit.partial")).mkdir(parents=True)
        (tmp_path / name / "run" / ("fit" if name.endswith("1") else "fit.partial") / "_KEY").write_text(key + "\n", encoding="utf-8")
        (tmp_path / name / "run_config.json").write_text(f'{{"seed": {seed}}}', encoding="utf-8")
    summary = {"quarantined": {}, "uncertified": {"gout": ["with_sv (outer_criterion_met)"]}}
    assert sv_gain_meta.partners_of("aou-p1") == ["aou-p2"] and sv_gain_meta.partners_of("aou-wl15") == []
    reasons, uncertified = sv_gain_meta.validity("aou-p1", ["aou-p2"], "gout", {"snv_indel": 1e-4, "with_sv": 2e-4}, summary)
    assert reasons == [] and uncertified == ["with_sv (outer_criterion_met)"]
    reasons, _ = sv_gain_meta.validity("aou-p1", ["aou-p2"], "gout", {"snv_indel": 1.6e-35, "with_sv": float("nan")}, summary)
    assert len(reasons) == 2 and all("null" in reason for reason in reasons)
    reasons, _ = sv_gain_meta.validity("aou-q1", ["aou-q2"], "ldl", {"snv_indel": 1.0, "with_sv": 1.0}, {"quarantined": {"ldl": ["with_sv (nonfinite)"]}})
    assert len(reasons) == 3 and "cannot be scored" in reasons[0] and "fit step key" in reasons[1] and "seed" in reasons[2]
    (tmp_path / "aou-p2" / "run" / "NULL_COLLAPSE.json").write_text('{"trait": "gout", "null_collapse": true, "why": "mean |b| 1e-21"}', encoding="utf-8")
    reasons, _ = sv_gain_meta.validity("aou-p1", ["aou-p2"], "gout", {"snv_indel": 1e-4, "with_sv": 2e-4}, summary)
    assert len(reasons) == 1 and "no genetic signal" in reasons[0] and "1e-21" in reasons[0]

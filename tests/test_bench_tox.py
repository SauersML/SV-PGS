"""bench-tox sealing, splits and power, on synthetic inputs only (no challenge data in tests)."""
import numpy as np
import pandas as pd
import pytest

from benchmarks.bench_tox import power, seal, splits


def synthetic_lines(generator):
    rows = []
    for population, continent in splits.CONTINENT.items():
        for family in range(12):
            size = 1 + int(generator.integers(0, 3))
            for member in range(size):
                rows.append({"line": f"{population}{family:02d}{member}", "family": f"{population}{family:02d}", "continent": continent})
    return pd.DataFrame(rows)


def test_sealing_depends_only_on_the_id_and_takes_about_a_quarter():
    ids = [f"NCGC{index:08d}.01" for index in range(4000)]
    development, confirmation = seal.split_compounds(ids)
    assert set(development) | set(confirmation) == set(ids) and not set(development) & set(confirmation)
    assert [seal.sealed(compound) for compound in ids] == [seal.sealed(compound) for compound in ids]
    share = len(confirmation) / len(ids)
    # Binomial(4000, 1/4): the share sits within 4 SDs of 1/4.
    assert abs(share - 0.25) < 4 * np.sqrt(0.25 * 0.75 / len(ids))


def test_random_folds_keep_families_and_kin_together_and_cover_every_line_once():
    generator = np.random.default_rng(3)
    lines = synthetic_lines(generator)
    related = [(lines["line"].iloc[0], lines["line"].iloc[-1])]
    folds = splits.random_folds(lines, related)
    tested = [line for fold in folds for line in fold["test"]]
    assert sorted(tested) == sorted(lines["line"])
    fold_of = {line: index for index, fold in enumerate(folds) for line in fold["test"]}
    for _, members in lines.groupby("family"):
        assert len({fold_of[line] for line in members["line"]}) == 1
    assert fold_of[related[0][0]] == fold_of[related[0][1]]
    assert folds == splits.random_folds(lines, related)


def test_leave_one_continent_out_holds_out_each_group_whole():
    lines = synthetic_lines(np.random.default_rng(4))
    held = splits.leave_one_continent_out(lines)
    assert sorted(split["name"] for split in held) == sorted(f"lococ/{group}" for group in set(splits.CONTINENT.values()))
    for split in splits.all_splits(lines):
        assert not set(split["test"]) & set(split["train"]) and set(split["test"]) | set(split["train"]) == set(lines["line"])


def test_null_r2_moments_match_simulation():
    generator = np.random.default_rng(5)
    lines = 160
    draws = [np.corrcoef(generator.standard_normal(lines), generator.standard_normal(lines))[0, 1] ** 2 for _ in range(20000)]
    mean, sd = power.null_r2_moments(lines)
    assert abs(np.mean(draws) - mean) < 4 * sd / np.sqrt(len(draws))
    assert abs(np.std(draws) - sd) < 0.05 * sd


def test_locus_power_is_monotone_and_the_detectable_share_attains_it():
    shares = np.linspace(0.001, 0.2, 50)
    values = power.locus_power(793, shares, 5e-8)
    assert np.all(np.diff(values) > 0)
    share = power.detectable_share(793, 5e-8, 0.8)
    assert power.locus_power(793, share, 5e-8) == pytest.approx(0.8, abs=1e-9)


def test_expected_polygenic_r2_limits():
    assert power.expected_polygenic_r2(0.5, 0, 6e4) == 0
    assert power.expected_polygenic_r2(0.5, 1e12, 6e4) == pytest.approx(0.5, rel=1e-6)

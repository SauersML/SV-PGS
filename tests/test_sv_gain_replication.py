import numpy as np
import pytest

from benchmarks.bench_real.sv_gain_replication import ALPHA, SCREENING_LEVEL, promising, replicate, report


def test_promising_screens_on_one_sided_p_and_a_valid_matched_pair():
    assert promising(0.043, matched_arms=True, identical_code=True)
    assert not promising(0.20, matched_arms=True, identical_code=True)
    assert not promising(0.001, matched_arms=True, identical_code=False)
    assert not promising(0.001, matched_arms=False, identical_code=True)
    assert not promising(SCREENING_LEVEL, matched_arms=True, identical_code=True)


def paired(rng, people, effect):
    """A binary trait whose with_sv arm carries ``effect`` of the liability the snv_indel arm misses."""
    covariates = np.column_stack([np.ones(people), rng.normal(size=people)])
    extra = rng.normal(size=people)
    baseline = rng.normal(size=people)
    candidate = baseline + effect * extra
    liability = baseline + effect * extra + 0.5 * covariates[:, 1] + rng.normal(size=people)
    return np.asarray(liability > 1.0, dtype=float), covariates, baseline, candidate


def test_a_real_gain_is_one_sided_significant_and_a_null_pair_is_not():
    rng = np.random.default_rng(0)
    people = 4000
    families = np.arange(people) // 2
    target, covariates, baseline, candidate = paired(rng, people, effect=0.5)
    found = replicate(target, covariates, families, baseline, candidate, seed=1, resamples=200, permutations=20)
    assert found["estimate"] > 0.0
    assert found["p_one_sided"] < ALPHA
    assert found["upper_95"] > found["estimate"]
    assert found["ci_low"] < found["estimate"] < found["ci_high"]
    assert 0.5 < found["null_z_sd"] < 2.0
    assert found["primary_p"] in (found["p_one_sided"], found["p_permutation_one_sided"])
    assert any("replicated" in line for line in report(found, "trait"))

    same = replicate(target, covariates, families, baseline, baseline.copy(), seed=1, resamples=200, permutations=0)
    assert same["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert same["p_one_sided"] != same["p_one_sided"] or same["p_one_sided"] >= ALPHA


def test_people_and_family_counts_come_from_the_rows_and_small_cells_are_masked():
    rng = np.random.default_rng(1)
    people = 300
    target, covariates, baseline, candidate = paired(rng, people, effect=0.0)
    found = replicate(target, covariates, np.arange(people) // 3, baseline, candidate, seed=2, resamples=50, permutations=0)
    assert found["people"] == people
    assert found["families"] == people // 3
    text = "\n".join(report({**found, "cases": 7}, "trait"))
    assert "<=20" in text and " 7" not in text

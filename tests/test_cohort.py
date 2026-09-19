"""The shared cohort: covariates with full rank, NaN-masked multi-trait targets, kinship-grouped folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.cohort import (
    SECOND_DEGREE_KINSHIP,
    AncestryPcs,
    build_cohort,
    indicator_columns,
    kinship_components,
    kinship_folds,
)


def _write_ancestry(path: Path, rows: list[tuple[str, list[float]]]) -> Path:
    lines = ["research_id\tancestry_pred\tpca_features"]
    lines += [f"{research_id}\teur\t{json.dumps(components)}" for research_id, components in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_ancestry_pcs_keep_every_component_and_follow_the_requested_order(tmp_path: Path) -> None:
    table = _write_ancestry(tmp_path / "ancestry.tsv", [("R1", [0.1, -0.2, 0.3]), ("R2", [0.4, 0.5, -0.6])])

    ancestry = AncestryPcs.read(table)

    np.testing.assert_array_equal(ancestry.for_samples(["R2", "R1"]), [[0.4, 0.5, -0.6], [0.1, -0.2, 0.3]])
    with pytest.raises(ValueError, match="1 samples have no genetic PCs"):
        ancestry.for_samples(["R1", "R3"])


def test_ragged_or_non_finite_pcs_fail_loudly(tmp_path: Path) -> None:
    ragged = _write_ancestry(tmp_path / "ragged.tsv", [("R1", [0.1, 0.2]), ("R2", [0.3])])
    with pytest.raises(ValueError):
        AncestryPcs.read(ragged)
    (tmp_path / "nan.tsv").write_text("research_id\tpca_features\nR1\t[NaN, 0.2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        AncestryPcs.read(tmp_path / "nan.tsv")


def test_indicator_columns_drop_exactly_one_level() -> None:
    kept, values = indicator_columns(["45878463", "45880669", "45878463", "1177221"])

    assert kept == ("45878463", "45880669")
    np.testing.assert_array_equal(values, [[1, 0], [0, 1], [1, 0], [0, 0]])
    assert indicator_columns(["only"])[1].shape == (1, 0)


def test_components_join_relatives_above_second_degree_and_ignore_outsiders() -> None:
    samples = ["A", "B", "C", "D", "E"]
    third_degree = SECOND_DEGREE_KINSHIP / 2

    components = kinship_components(
        samples,
        first_ids=["A", "B", "D", "E"],
        second_ids=["B", "C", "E", "X"],
        kinship=[0.5, 0.25, third_degree, 0.5],
    )

    assert components[0] == components[1] == components[2]
    assert len({components[0], components[3], components[4]}) == 3


def test_folds_keep_components_whole_and_balance_each_stratum() -> None:
    rng = np.random.default_rng(7)
    sample_count = 600
    samples = [f"S{index}" for index in range(sample_count)]
    strata = np.array(["imputed|eur", "imputed|afr", "long_read|eur"])[rng.integers(0, 3, sample_count)]
    # Families of 2-4 relatives, each inside one stratum.
    first_ids: list[str] = []
    second_ids: list[str] = []
    for start in range(0, 240, 4):
        family = list(range(start, start + int(rng.integers(2, 5))))
        strata[family] = strata[family[0]]
        first_ids += [samples[family[0]]] * (len(family) - 1)
        second_ids += [samples[member] for member in family[1:]]
    components = kinship_components(samples, first_ids, second_ids, [0.25] * len(first_ids))

    folds = kinship_folds(components, list(strata), fold_count=5, seed=11)

    for component in np.unique(components):
        assert np.unique(folds[components == component]).shape == (1,)
    for stratum in np.unique(strata):
        in_stratum = strata == stratum
        counts = np.bincount(folds[in_stratum], minlength=5)
        largest = np.bincount(components[in_stratum]).max()
        assert counts.max() - counts.min() <= largest


def test_folds_follow_sample_identities_not_the_order_they_are_listed_in() -> None:
    rng = np.random.default_rng(13)
    samples = [f"S{index:03d}" for index in range(200)]
    strata = dict(zip(samples, np.array(["imputed|eur", "long_read|afr"])[rng.integers(0, 2, 200)], strict=True))
    first_ids = [samples[index] for index in range(0, 60, 3)]
    second_ids = [samples[index + 1] for index in range(0, 60, 3)]
    for first, second in zip(first_ids, second_ids, strict=True):
        strata[second] = strata[first]

    def folds_by_id(order: list[str]) -> dict[str, int]:
        pair_order = rng.permutation(len(first_ids))
        components = kinship_components(
            order,
            [first_ids[index] for index in pair_order],
            [second_ids[index] for index in pair_order],
            [0.25] * len(first_ids),
        )
        folds = kinship_folds(components, [strata[sample] for sample in order], fold_count=5, seed=3)
        return dict(zip(order, folds.tolist(), strict=True))

    listed = folds_by_id(samples)
    for _ in range(3):
        assert folds_by_id(list(rng.permutation(samples))) == listed


def test_fold_assignment_is_reproducible_from_its_seed() -> None:
    components = np.arange(50, dtype=np.int64)
    strata = ["one"] * 50

    first = kinship_folds(components, strata, fold_count=5, seed=3)

    np.testing.assert_array_equal(first, kinship_folds(components, strata, fold_count=5, seed=3))
    assert not np.array_equal(first, kinship_folds(components, strata, fold_count=5, seed=4))
    with pytest.raises(ValueError, match="at least two folds"):
        kinship_folds(components, strata, fold_count=1, seed=3)


def _ancestry(research_ids: list[str], rng: np.random.Generator) -> AncestryPcs:
    return AncestryPcs(research_ids=tuple(research_ids), components=rng.standard_normal((len(research_ids), 2)))


def test_build_cohort_assembles_full_rank_covariates_and_masked_targets() -> None:
    rng = np.random.default_rng(5)
    research_ids = [f"R{index}" for index in range(8)]
    ages = rng.uniform(20, 80, 8)

    cohort = build_cohort(
        research_ids,
        person_covariates={"age": ages, "age_squared": ages**2},
        categorical_covariates={"sex_at_birth": ["F", "M"] * 4},
        ancestry=_ancestry(research_ids[::-1], rng),
        pipeline_half=["h0", "h0", "h1", "h1", "h0", "h1", "h0", "h1"],
        genotype_source=["imputed"] * 6 + ["long_read"] * 2,
        trait_targets={"ldl": {"R0": 1.5, "R3": -0.5}, "t2d": {research_id: 1.0 for research_id in research_ids}},
    )

    assert cohort.covariate_names == (
        "intercept", "age", "age_squared", "sex_at_birth=M", "pipeline_half=h1", "genotype_source=long_read",
        "PC1", "PC2",
    )
    assert cohort.covariates.shape == (8, 8)
    assert cohort.trait_names == ("ldl", "t2d")
    np.testing.assert_array_equal(cohort.observed[:, 0], [True, False, False, True, False, False, False, False])
    assert cohort.observed[:, 1].all()
    assert cohort.targets[3, 0] == -0.5


def test_build_cohort_rejects_collinear_covariates_and_missing_values() -> None:
    rng = np.random.default_rng(9)
    research_ids = [f"R{index}" for index in range(6)]
    ages = rng.uniform(20, 80, 6)
    common = dict(
        categorical_covariates={},
        ancestry=_ancestry(research_ids, rng),
        pipeline_half=["h0"] * 6,
        genotype_source=["imputed"] * 6,
        trait_targets={"ldl": {"R0": 1.0}},
    )

    with pytest.raises(ValueError, match="rank-deficient"):
        build_cohort(research_ids, person_covariates={"age": ages, "age_again": 2 * ages}, **common)
    with pytest.raises(ValueError, match="missing values"):
        build_cohort(research_ids, person_covariates={"age": np.where(ages > 50, np.nan, ages)}, **common)
    with pytest.raises(ValueError, match="non-finite target"):
        build_cohort(
            research_ids, person_covariates={"age": ages}, **{**common, "trait_targets": {"ldl": {"R0": np.nan}}}
        )


def test_build_cohort_rejects_a_trait_keyed_by_ids_outside_the_cohort() -> None:
    rng = np.random.default_rng(3)
    research_ids = [str(1000 + index) for index in range(6)]

    with pytest.raises(ValueError, match="'ldl' has no target for any cohort sample"):
        build_cohort(
            research_ids,
            person_covariates={"age": rng.uniform(20, 80, 6)},
            categorical_covariates={},
            ancestry=_ancestry(research_ids, rng),
            pipeline_half=["h0"] * 6,
            genotype_source=["imputed"] * 6,
            trait_targets={"ldl": {1000 + index: 1.0 for index in range(6)}},
        )

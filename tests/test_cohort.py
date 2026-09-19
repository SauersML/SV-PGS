"""The shared cohort: covariates with full rank, NaN-masked multi-trait targets, kinship-grouped folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.cohort import (
    DUPLICATE_KINSHIP,
    SECOND_DEGREE_KINSHIP,
    AncestryPcs,
    build_cohort,
    indicator_columns,
    kinship_components,
    kinship_folds,
    resolve_cohort_rows,
)
from sv_pgs.dosage_store import HalfSamples
from sv_pgs.sample_crosswalk import SampleCrosswalk
from sv_pgs.sample_ids import ResearchId, SequencingId


def _write_ancestry(path: Path, rows: list[tuple[str, list[float]]]) -> Path:
    lines = ["research_id\tancestry_pred\tpca_features"]
    lines += [f"{research_id}\teur\t{json.dumps(components)}" for research_id, components in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_ancestry_pcs_keep_every_component_and_follow_the_requested_order(tmp_path: Path) -> None:
    table = _write_ancestry(tmp_path / "ancestry.tsv", [("R1", [0.1, -0.2, 0.3]), ("R2", [0.4, 0.5, -0.6])])

    ancestry = AncestryPcs.read(table)

    np.testing.assert_array_equal(
        ancestry.for_samples([ResearchId("R2"), ResearchId("R1")]), [[0.4, 0.5, -0.6], [0.1, -0.2, 0.3]]
    )
    with pytest.raises(ValueError, match="1 samples have no genetic PCs"):
        ancestry.for_samples([ResearchId("R1"), ResearchId("R3")])
    with pytest.raises(TypeError, match="keyed by ResearchId"):
        ancestry.for_samples([SequencingId("R1")])


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


def _ancestry(research_ids: list[ResearchId], rng: np.random.Generator) -> AncestryPcs:
    return AncestryPcs(
        research_ids=tuple(research_id.value for research_id in research_ids),
        components=rng.standard_normal((len(research_ids), 2)),
    )


def test_build_cohort_assembles_full_rank_covariates_and_masked_targets() -> None:
    rng = np.random.default_rng(5)
    research_ids = [ResearchId(f"R{index}") for index in range(8)]
    ages = rng.uniform(20, 80, 8)

    cohort = build_cohort(
        research_ids,
        person_covariates={"age": ages, "age_squared": ages**2},
        categorical_covariates={"sex_at_birth": ["F", "M"] * 4},
        ancestry=_ancestry(research_ids[::-1], rng),
        pipeline_half=["h0", "h0", "h1", "h1", "h0", "h1", "h0", "h1"],
        genotype_source=["imputed"] * 6 + ["long_read"] * 2,
        trait_targets={"ldl": {"R0": 1.5, "R3": -0.5}, "t2d": {research_id.value: 1.0 for research_id in research_ids}},
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
    research_ids = [ResearchId(f"R{index}") for index in range(6)]
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
    research_ids = [ResearchId(str(1000 + index)) for index in range(6)]

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




def _crosswalk(pairs: dict[str, str]) -> SampleCrosswalk:
    """Sequencing name -> research ID."""
    return SampleCrosswalk(research_ids=tuple(pairs.values()), sequencing_ids=tuple(pairs))


def _research(*values: str) -> list[ResearchId]:
    return [ResearchId(value) for value in values]


def _imputed(*names: str) -> HalfSamples:
    return HalfSamples("dragen_sample", names)


def _long_read(*names: str) -> HalfSamples:
    return HalfSamples("research_id", names)


def test_a_participant_in_both_halves_keeps_only_the_truth_row() -> None:
    rows = resolve_cohort_rows(
        [_imputed("D1", "D3"), _long_read("R1", "R2")],
        crosswalk=_crosswalk({"D1": "R1", "D3": "R3"}),
        kinship_pairs=[],
    )

    assert rows.research_ids == tuple(_research("R3", "R1", "R2"))
    assert rows.genotype_source == ("imputed", "long_read", "long_read")
    assert (rows.store_half, rows.store_column) == ((0, 1, 1), (1, 0, 1))
    assert rows.imputed_rows_sharing_a_research_id == 1


def test_names_that_collide_across_the_halves_are_different_people() -> None:
    # The DRAGEN name "1001" is spelled like the truth participant 1001, but the crosswalk
    # maps it to participant 2002, and the KING pair links 2002 (not 1001) to 3003.
    rows = resolve_cohort_rows(
        [_imputed("1001"), _long_read("1001", "3003")],
        crosswalk=_crosswalk({"1001": "2002"}),
        kinship_pairs=[(SequencingId("1001"), ResearchId("3003"), 0.25)],
    )

    assert rows.research_ids == tuple(_research("2002", "1001", "3003"))
    assert rows.imputed_rows_sharing_a_research_id == 0
    assert rows.kinship_pairs == ((ResearchId("2002"), ResearchId("3003"), 0.25),)


def test_ids_of_different_namespaces_never_compare() -> None:
    with pytest.raises(TypeError, match="compared a ResearchId with a SequencingId"):
        _ = ResearchId("1001") == SequencingId("1001")
    with pytest.raises(TypeError):
        _ = {ResearchId("1001"), SequencingId("1001")}
    with pytest.raises(TypeError):
        _ = ResearchId("1001") == "1001"
    assert ResearchId("1001") == ResearchId("1001") and ResearchId("1001") != ResearchId("1002")


def test_kinship_pairs_must_name_typed_samples() -> None:
    with pytest.raises(TypeError, match="not a typed sample ID"):
        resolve_cohort_rows([_imputed("D1"), _long_read("R1")], _crosswalk({"D1": "R2"}), [("R1", "D1", 0.25)])


def test_a_duplicate_genome_across_halves_keeps_the_truth_row_but_twins_within_a_half_stay() -> None:
    duplicate = DUPLICATE_KINSHIP * 1.4
    rows = resolve_cohort_rows(
        [_imputed("D4", "D5"), _imputed("D6", "D7"), _long_read("R1", "R2")],
        crosswalk=_crosswalk({"D4": "R4", "D5": "R5", "D6": "R6", "D7": "R7"}),
        # D4 is R1's genome under another research ID; D5/D6 (across the two imputed
        # halves) and R1/R2 are twins in one namespace; D7 is R2's first-degree relative.
        kinship_pairs=[
            (SequencingId("D4"), ResearchId("R1"), duplicate),
            (SequencingId("D5"), SequencingId("D6"), duplicate),
            (ResearchId("R1"), ResearchId("R2"), duplicate),
            (ResearchId("R2"), SequencingId("D7"), 0.25),
        ],
    )

    assert rows.research_ids == tuple(_research("R5", "R6", "R7", "R1", "R2"))
    assert rows.store_half == (0, 1, 1, 2, 2)
    assert rows.imputed_rows_duplicating_a_truth_genome == 1
    assert rows.imputed_rows_sharing_a_research_id == 0


@pytest.mark.parametrize(
    ("halves", "crosswalk", "pairs", "message"),
    [
        ([_long_read("R1"), _long_read("R1")], {}, [], "truth halves list a research ID more than once"),
        ([_imputed("D2"), _imputed("D2")], {"D2": "R2"}, [], "two imputed halves list the same sequencing sample"),
        ([_imputed("D2", "D3")], {"D2": "R2"}, [], "1 store samples have no crosswalk row"),
        ([_imputed("D2"), _long_read("R1")], {"D2": "R2", "D9": "R9"}, [("D9", "R1")], "no store half lists"),
    ],
)
def test_repeats_and_unmapped_samples_are_errors(halves, crosswalk, pairs, message) -> None:
    typed_pairs = [(SequencingId(first), ResearchId(second), 0.25) for first, second in pairs]
    with pytest.raises(ValueError, match=message):
        resolve_cohort_rows(halves, _crosswalk(crosswalk), typed_pairs)


def test_replaced_row_counts_of_one_to_twenty_are_suppressed_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    def resolve(overlap: int) -> None:
        crosswalk = _crosswalk({f"D{index}": f"R{index}" for index in range(overlap)})
        resolve_cohort_rows([_imputed(*crosswalk.sequencing_ids), _long_read(*crosswalk.research_ids)], crosswalk, [])

    with caplog.at_level("INFO", logger="sv_pgs.cohort"):
        resolve(3)
        resolve(25)

    first, second = (record.getMessage() for record in caplog.records)
    assert "1-20 (suppressed) sharing a research ID" in first and "3" not in first
    assert "25 sharing a research ID" in second


def test_relatives_across_the_two_halves_share_a_fold() -> None:
    rng = np.random.default_rng(21)
    names = [f"{index:03d}" for index in range(150)]
    # The truth participants and the imputed samples use the same strings on purpose:
    # the crosswalk maps each imputed name to another participant.
    rows = resolve_cohort_rows(
        [_long_read(*names), _imputed(*names)],
        _crosswalk({name: f"P{name}" for name in names}),
        [(ResearchId(names[index]), SequencingId(names[index + 1]), 0.25) for index in range(0, 148, 2)],
    )
    ancestry = dict(zip(rows.research_ids, np.array(["eur", "afr"])[rng.integers(0, 2, len(rows.research_ids))], strict=True))
    for first, second, _coefficient in rows.kinship_pairs:
        ancestry[second] = ancestry[first]
    components = kinship_components(
        rows.research_ids,
        [first for first, _second, _coefficient in rows.kinship_pairs],
        [second for _first, second, _coefficient in rows.kinship_pairs],
        [coefficient for _first, _second, coefficient in rows.kinship_pairs],
    )
    strata = [f"{source}|{ancestry[research_id]}" for research_id, source in zip(rows.research_ids, rows.genotype_source, strict=True)]

    folds = dict(zip(rows.research_ids, kinship_folds(components, strata, fold_count=5, seed=2).tolist(), strict=True))

    assert len(rows.research_ids) == 300
    assert all(folds[first] == folds[second] for first, second, _coefficient in rows.kinship_pairs)

def test_build_cohort_rejects_a_participant_listed_twice_or_an_untyped_id() -> None:
    rng = np.random.default_rng(4)
    common = dict(
        person_covariates={"age": rng.uniform(20, 80, 4)},
        categorical_covariates={},
        ancestry=_ancestry(_research("R0", "R1", "R2"), rng),
        pipeline_half=["h0"] * 4,
        genotype_source=["long_read", "long_read", "imputed", "imputed"],
        trait_targets={"ldl": {"R0": 1.0}},
    )

    with pytest.raises(ValueError, match="repeats a participant"):
        build_cohort(_research("R0", "R1", "R1", "R2"), **common)
    with pytest.raises(TypeError, match="keyed by ResearchId"):
        build_cohort(["R0", "R1", "R2", "R3"], **common)

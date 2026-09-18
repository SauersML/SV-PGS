"""Store sources join samples through the CDR crosswalk, never by name."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sv_pgs.sample_crosswalk import (
    ABSENT_COLUMN,
    SampleCrosswalk,
    source_columns_for_store_samples,
    source_sample_covariates,
)


def test_join_follows_the_crosswalk_and_never_matches_equal_names() -> None:
    crosswalk = SampleCrosswalk(research_ids=("R1", "R2", "R4"), sequencing_ids=("D1", "D2", "D4"))
    # "D3" is spelled the same in both lists but has no crosswalk row, and
    # "R4" is in the crosswalk but not in the source: neither may join.
    store_sequencing_ids = ["D1", "D2", "D3", "D4"]
    source_research_ids = ["R2", "R1", "D3"]

    columns = source_columns_for_store_samples(store_sequencing_ids, source_research_ids, crosswalk)

    np.testing.assert_array_equal(columns, [1, 0, ABSENT_COLUMN, ABSENT_COLUMN])


def test_name_equal_ids_do_not_join_without_a_crosswalk_row() -> None:
    crosswalk = SampleCrosswalk(research_ids=("R9",), sequencing_ids=("D9",))
    shared_names = ["S1", "S2", "S3"]

    columns = source_columns_for_store_samples(shared_names, shared_names, crosswalk)

    assert (columns == ABSENT_COLUMN).all()


@pytest.mark.parametrize(
    ("research_ids", "sequencing_ids", "message"),
    [
        (("R1", "R1"), ("D1", "D2"), "repeats a research ID"),
        (("R1", "R2"), ("D1", "D1"), "repeats a sequencing ID"),
        (("R1", ""), ("D1", "D2"), "blank research ID"),
        (("R1",), ("D1", "D2"), "one sequencing ID per research ID"),
    ],
)
def test_crosswalk_must_be_one_to_one(research_ids: tuple[str, ...], sequencing_ids: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SampleCrosswalk(research_ids=research_ids, sequencing_ids=sequencing_ids)


def test_source_with_repeated_research_ids_is_rejected() -> None:
    crosswalk = SampleCrosswalk(research_ids=("R1",), sequencing_ids=("D1",))
    with pytest.raises(ValueError, match="repeats a research ID"):
        source_columns_for_store_samples(["D1"], ["R1", "R1"], crosswalk)


def test_crosswalk_reads_a_delimited_table(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.tsv"
    path.write_text("person_id\tdragen_id\nR1\tD1\nR2\tD2\n", encoding="utf-8")

    crosswalk = SampleCrosswalk.read(path, research_id_column="person_id", sequencing_id_column="dragen_id")

    assert crosswalk == SampleCrosswalk(research_ids=("R1", "R2"), sequencing_ids=("D1", "D2"))


def test_availability_and_no_call_rate_covariates() -> None:
    columns = np.array([1, ABSENT_COLUMN, 0], dtype=np.int64)
    source_rates = np.array([0.02, 0.1])

    available, no_call_rate = source_sample_covariates(columns, source_rates)

    assert available.tolist() == [True, False, True]
    np.testing.assert_allclose(no_call_rate, [0.1, 0.0, 0.02])

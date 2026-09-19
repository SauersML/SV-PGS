"""Store sources join samples through the CDR crosswalk, never by name."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sv_pgs.dosage_store import HalfSamples
from sv_pgs.sample_crosswalk import (
    ABSENT_COLUMN,
    SampleCrosswalk,
    source_columns_for_store_samples,
    store_research_ids,
)


def test_join_follows_the_crosswalk_and_never_matches_equal_names() -> None:
    crosswalk = SampleCrosswalk(research_ids=("R1", "R2", "R4"), sequencing_ids=("D1", "D2", "D4"))
    # "D3" is spelled the same in both lists but has no crosswalk row, and
    # "R4" is in the crosswalk but not in the source: neither may join.
    store_sequencing_ids = ["D1", "D2", "D3", "D4"]
    source_research_ids = ["R2", "R1", "D3"]

    columns = source_columns_for_store_samples(HalfSamples("dragen_sample", tuple(store_sequencing_ids)), source_research_ids, crosswalk)

    np.testing.assert_array_equal(columns, [1, 0, ABSENT_COLUMN, ABSENT_COLUMN])


def test_name_equal_ids_do_not_join_without_a_crosswalk_row() -> None:
    crosswalk = SampleCrosswalk(research_ids=("R9",), sequencing_ids=("D9",))
    shared_names = ["S1", "S2", "S3"]

    columns = source_columns_for_store_samples(HalfSamples("dragen_sample", tuple(shared_names)), shared_names, crosswalk)

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
        source_columns_for_store_samples(HalfSamples("dragen_sample", ("D1",)), ["R1", "R1"], crosswalk)


def test_crosswalk_reads_a_delimited_table(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.tsv"
    path.write_text("person_id\tdragen_id\nR1\tD1\nR2\tD2\n", encoding="utf-8")

    crosswalk = SampleCrosswalk.read(path, research_id_column="person_id", sequencing_id_column="dragen_id")

    assert crosswalk == SampleCrosswalk(research_ids=("R1", "R2"), sequencing_ids=("D1", "D2"))



def test_store_research_ids_map_a_half_and_refuse_gaps_and_repeats() -> None:
    crosswalk = SampleCrosswalk(research_ids=("R1", "R2", "R3"), sequencing_ids=("D1", "D2", "D3"))

    assert store_research_ids(HalfSamples("dragen_sample", ("D3", "D1")), crosswalk) == ("R3", "R1")
    with pytest.raises(ValueError, match="no crosswalk row"):
        store_research_ids(HalfSamples("dragen_sample", ("D1", "D9")), crosswalk)
    with pytest.raises(ValueError, match="more than once"):
        HalfSamples("dragen_sample", ("D2", "D2"))


def test_a_half_named_by_research_id_is_never_looked_up_among_sequencing_names() -> None:
    # A long-read participant whose research ID is spelled like another person's DRAGEN name:
    # a name lookup would join the wrong person, so the research-ID half is refused outright.
    crosswalk = SampleCrosswalk(research_ids=("R1", "R2"), sequencing_ids=("1234567", "D2"))
    long_read_half = HalfSamples("research_id", ("1234567",))

    with pytest.raises(ValueError, match="never compared across namespaces"):
        store_research_ids(long_read_half, crosswalk)
    with pytest.raises(ValueError, match="never compared across namespaces"):
        source_columns_for_store_samples(long_read_half, ["R1", "R2"], crosswalk)

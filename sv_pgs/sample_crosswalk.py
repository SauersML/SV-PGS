"""Join genotype sources to the store's samples through the CDR crosswalk.

The imputed BCFs name samples by their DRAGEN sequencing IDs; the GATK-SV call
set, phenotypes and covariates use AoU research IDs. The two namespaces are
joined only through the program's crosswalk, a one-to-one table of
(research ID, sequencing ID) pairs. Two IDs are never matched because their
strings are equal: an ID that happens to appear in both namespaces but has no
crosswalk row is not joined.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sv_pgs._typing import I64Array
from sv_pgs.dosage_store import HalfSamples

ABSENT_COLUMN = -1


@dataclass(frozen=True, slots=True)
class SampleCrosswalk:
    """One-to-one pairs of research IDs and sequencing IDs."""

    research_ids: tuple[str, ...]
    sequencing_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.research_ids) != len(self.sequencing_ids):
            raise ValueError("the crosswalk needs one sequencing ID per research ID.")
        for label, identifiers in (("research", self.research_ids), ("sequencing", self.sequencing_ids)):
            if not all(identifiers):
                raise ValueError(f"the crosswalk has a blank {label} ID.")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(
                    f"the crosswalk repeats a {label} ID; resolve duplicate samples before joining."
                )

    @classmethod
    def read(cls, path: str | Path, research_id_column: str, sequencing_id_column: str) -> SampleCrosswalk:
        """Read the crosswalk from a tab- or comma-delimited table with a header."""
        with Path(path).open(newline="", encoding="utf-8") as handle:
            dialect = csv.Sniffer().sniff(handle.readline(), delimiters="\t,")
            handle.seek(0)
            rows = list(csv.DictReader(handle, dialect=dialect))
        return cls(
            research_ids=tuple(row[research_id_column].strip() for row in rows),
            sequencing_ids=tuple(row[sequencing_id_column].strip() for row in rows),
        )


def _sequencing_names(half: HalfSamples) -> tuple[str, ...]:
    """The half's names, which the crosswalk can map only when they are sequencing names."""
    if half.namespace != "dragen_sample":
        raise ValueError(
            f"the crosswalk maps DRAGEN sequencing names; this half is named by {half.namespace}, and names are "
            "never compared across namespaces."
        )
    return half.names


def source_columns_for_store_samples(
    store_half: HalfSamples,
    source_research_ids: Sequence[str],
    crosswalk: SampleCrosswalk,
) -> I64Array:
    """For each sample of an imputed store half, its column in a research-ID-keyed source, or ABSENT_COLUMN.

    A store sample is absent when the crosswalk has no row for it or the source does not carry
    its participant.
    """
    store_sequencing_ids = _sequencing_names(store_half)
    if len(set(source_research_ids)) != len(source_research_ids):
        raise ValueError("the genotype source repeats a research ID.")
    research_of_sequencing = dict(zip(crosswalk.sequencing_ids, crosswalk.research_ids))
    column_of_research = {research_id: column for column, research_id in enumerate(source_research_ids)}
    columns = np.full(len(store_sequencing_ids), ABSENT_COLUMN, dtype=np.int64)
    for store_index, sequencing_id in enumerate(store_sequencing_ids):
        research_id = research_of_sequencing.get(sequencing_id)
        if research_id is not None:
            columns[store_index] = column_of_research.get(research_id, ABSENT_COLUMN)
    return columns


def store_research_ids(store_half: HalfSamples, crosswalk: SampleCrosswalk) -> tuple[str, ...]:
    """The research ID of each sample of an imputed store half, in store column order.

    Fails on a sample the crosswalk has no row for. A half named by research ID (the long-read
    half) is refused: its names are already research IDs, and looking them up among sequencing
    names would join any that collide by chance to the wrong person.
    """
    half_sequencing_ids = _sequencing_names(store_half)
    research_of_sequencing = dict(zip(crosswalk.sequencing_ids, crosswalk.research_ids))
    unmapped = [sequencing_id for sequencing_id in half_sequencing_ids if sequencing_id not in research_of_sequencing]
    if unmapped:
        raise ValueError(f"{len(unmapped)} store samples have no crosswalk row.")
    return tuple(research_of_sequencing[sequencing_id] for sequencing_id in half_sequencing_ids)

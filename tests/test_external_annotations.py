"""External SV/VNTR catalogue maps: tiers, one-to-one assignment, VNTR motif join, payload."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.external_annotations import (
    TIER_COORDINATE,
    TIER_EXACT,
    TIER_SEQUENCE,
    changed_sequences,
    majority_motif_length,
    match_records,
    match_tr_loci,
    squared_z,
)

_GENERATOR = np.random.default_rng(5)
_INSERTED = "".join(_GENERATOR.choice(list("ACGT"), 300))
_DELETED = "".join(_GENERATOR.choice(list("ACGT"), 400))
_OTHER = "".join(_GENERATOR.choice(list("ACGT"), 300))


def _records(rows):
    identifiers, chromosomes, positions, refs, alts = zip(*rows)
    return changed_sequences(list(identifiers), list(chromosomes), np.asarray(positions), list(refs), list(alts))


def test_changed_sequences_keep_deletions_and_insertions_only() -> None:
    records = _records(
        [
            ("snv", "chr1", 100, "A", "G"),
            ("del", "chr1", 1_000, "T" + _DELETED, "T"),
            ("small", "chr1", 2_000, "C", "CAAAA"),
            ("ins", "chr1", 3_000, "G", "G" + _INSERTED),
        ]
    )

    assert records.identifiers == ("del", "ins")
    assert records.rows.tolist() == [1, 3]
    assert records.kinds.tolist() == ["DEL", "INS"]
    assert records.starts.tolist() == [1_000, 3_000]
    assert records.ends.tolist() == [1_400, 3_000]
    assert records.sequences == (_DELETED, _INSERTED)


def test_match_records_tiers_and_one_to_one() -> None:
    panel = _records(
        [
            ("p_del", "chr1", 1_000, "T" + _DELETED, "T"),
            ("p_ins", "chr1", 3_000, "G", "G" + _INSERTED),
            ("p_other", "chr1", 6_000, "C", "C" + _OTHER),
            ("p_lonely", "chr2", 9_000, "A", "A" + _INSERTED),
        ]
    )
    external = _records(
        [
            ("e_exact", "chr1", 1_000, "T" + _DELETED, "T"),
            # Same insertion 40 bp away with a few edits: sequence tier.
            ("e_similar", "chr1", 3_040, "G", "G" + _INSERTED[:280] + "TTTTT" + _INSERTED[285:]),
            # A second claim on p_ins, weaker (coordinate-only sequence): displaced.
            ("e_rival", "chr1", 3_010, "G", "G" + _OTHER),
            # Different sequence at p_other: coordinate tier.
            ("e_coordinate", "chr1", 6_020, "C", "C" + _INSERTED),
            ("e_far", "chr1", 50_000, "G", "G" + _INSERTED),
        ]
    )

    matches = match_records(external, panel)

    assert matches.store_rows.tolist() == [0, 1, 2]
    assert matches.external_identifiers == ("e_exact", "e_similar", "e_coordinate")
    assert matches.tiers.tolist() == [TIER_EXACT, TIER_SEQUENCE, TIER_COORDINATE]
    assert matches.displaced_pairs == 1


def test_majority_motif_length_weighs_overlapping_bases() -> None:
    motif = majority_motif_length(
        np.array([100, 500]),
        np.array([200, 600]),
        np.array([90, 150, 1_000]),
        np.array([140, 170, 1_100]),
        np.array([3, 6, 4]),
    )

    # Locus 0: period 3 covers 40 bases, period 6 covers 20; locus 1 has no repeat.
    assert motif.tolist() == [3, 0]


def test_match_tr_loci_allow_trf_period_multiples() -> None:
    matches = match_tr_loci(
        ["e_double", "e_mismatch", "e_rival"],
        np.array([100, 500, 120]),
        np.array([200, 600, 150]),
        np.array([30, 7, 15]),
        np.array([110, 480]),
        np.array([210, 620]),
        np.array([15, 15]),
    )

    # e_double (motif 30 = 2 x 15) overlaps locus 0 by 90 bases and beats e_rival (30 bases);
    # e_mismatch's motif 7 agrees with no locus.
    assert matches.loci.tolist() == [0]
    assert matches.external_identifiers == ("e_double",)


def test_squared_z_is_scale_and_orientation_free() -> None:
    np.testing.assert_allclose(squared_z(np.array([0.2, -0.2]), np.array([0.1, 0.1])), [4.0, 4.0])
    with pytest.raises(ValueError, match="positive standard errors"):
        squared_z(np.array([0.1]), np.array([0.0]))

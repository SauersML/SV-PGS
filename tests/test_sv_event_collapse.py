"""Tests for :mod:`sv_pgs.sv_event_collapse`."""

from __future__ import annotations

from sv_pgs.config import VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.sv_event_collapse import (
    cluster_sv_events,
    select_representative_indices,
)


def _deletion(
    variant_id: str,
    *,
    chromosome: str = "1",
    position: int,
    length: float,
    training_support: int = 100,
) -> VariantRecord:
    return VariantRecord(
        variant_id=variant_id,
        variant_class=VariantClass.DELETION_LONG,
        chromosome=chromosome,
        position=position,
        length=length,
        allele_frequency=0.01,
        quality=1.0,
        training_support=training_support,
    )


def _snv(variant_id: str, *, position: int) -> VariantRecord:
    return VariantRecord(
        variant_id=variant_id,
        variant_class=VariantClass.SNV,
        chromosome="1",
        position=position,
        length=1.0,
        allele_frequency=0.2,
        quality=1.0,
    )


def test_three_near_duplicate_deletions_collapse_to_single_cluster() -> None:
    # Three deletions on chr1 around position 1_000_000 with 95%+ reciprocal
    # overlap on a ~1000bp event — should all collapse into a single cluster.
    records = [
        _deletion("DEL_a", position=1_000_000, length=1_000.0, training_support=80),
        _deletion("DEL_b", position=1_000_020, length=990.0, training_support=100),
        _deletion("DEL_c", position=1_000_010, length=1_005.0, training_support=95),
    ]
    cluster_map = cluster_sv_events(records)
    assert len(cluster_map) == 1
    [(representative_index, member_indices)] = cluster_map.items()
    assert sorted(member_indices) == [0, 1, 2]
    # Representative is the one with the highest training_support (index 1).
    assert representative_index == 1


def test_non_overlapping_same_chrom_same_svtype_do_not_cluster() -> None:
    records = [
        _deletion("DEL_left", position=100_000, length=500.0, training_support=80),
        _deletion("DEL_right", position=900_000, length=500.0, training_support=80),
    ]
    cluster_map = cluster_sv_events(records)
    assert cluster_map == {0: [0], 1: [1]}


def test_snv_records_never_cluster_even_at_same_position() -> None:
    records = [
        _snv("rs1", position=12_345),
        _snv("rs2", position=12_345),
        _snv("rs3", position=12_346),
    ]
    cluster_map = cluster_sv_events(records)
    assert cluster_map == {0: [0], 1: [1], 2: [2]}


def test_support_mismatch_blocks_clustering() -> None:
    # Geometrically identical deletions but with wildly different support
    # counts should not be considered the same event.
    records = [
        _deletion("DEL_lo", position=1_000_000, length=1_000.0, training_support=10),
        _deletion("DEL_hi", position=1_000_005, length=1_000.0, training_support=500),
    ]
    cluster_map = cluster_sv_events(records, support_count_ratio_tolerance=0.2)
    assert cluster_map == {0: [0], 1: [1]}


def test_different_svtype_blocks_clustering() -> None:
    # Same chromosome and overlapping position, but a deletion vs a
    # duplication — different SVTYPE => never collapsed.
    records = [
        VariantRecord(
            variant_id="DEL",
            variant_class=VariantClass.DELETION_LONG,
            chromosome="1",
            position=1_000_000,
            length=1_000.0,
            allele_frequency=0.01,
            training_support=100,
        ),
        VariantRecord(
            variant_id="DUP",
            variant_class=VariantClass.DUPLICATION_LONG,
            chromosome="1",
            position=1_000_005,
            length=1_000.0,
            allele_frequency=0.01,
            training_support=100,
        ),
    ]
    cluster_map = cluster_sv_events(records)
    assert cluster_map == {0: [0], 1: [1]}


def test_different_chromosome_blocks_clustering() -> None:
    records = [
        _deletion("DEL_chr1", chromosome="1", position=1_000_000, length=1_000.0),
        _deletion("DEL_chr2", chromosome="2", position=1_000_000, length=1_000.0),
    ]
    cluster_map = cluster_sv_events(records)
    assert cluster_map == {0: [0], 1: [1]}


def test_single_bp_event_does_not_crash_and_clusters_when_identical() -> None:
    # Single-bp breakend signals should still produce a valid span and
    # cluster when they describe the same site.
    records = [
        _deletion("BND_a", position=500_000, length=1.0, training_support=50),
        _deletion("BND_b", position=500_000, length=1.0, training_support=50),
    ]
    cluster_map = cluster_sv_events(records)
    assert len(cluster_map) == 1
    [(_, member_indices)] = cluster_map.items()
    assert sorted(member_indices) == [0, 1]


def test_mixed_input_preserves_snv_singletons_and_collapses_sv() -> None:
    records = [
        _snv("rs1", position=10),
        _deletion("DEL_a", position=1_000_000, length=1_000.0, training_support=80),
        _deletion("DEL_b", position=1_000_010, length=1_005.0, training_support=100),
        _snv("rs2", position=2_000_000),
    ]
    cluster_map = cluster_sv_events(records)
    representative_indices = select_representative_indices(cluster_map)
    # Three clusters: SNV0, SV-cluster (rep=2), SNV3.
    assert representative_indices == [0, 2, 3]
    assert cluster_map[0] == [0]
    assert sorted(cluster_map[2]) == [1, 2]
    assert cluster_map[3] == [3]


def test_deterministic_for_sorted_input() -> None:
    # Repeatable runs over the same input yield identical mappings.
    records = [
        _deletion("DEL_a", position=1_000_000, length=1_000.0, training_support=80),
        _deletion("DEL_b", position=1_000_020, length=990.0, training_support=100),
        _deletion("DEL_c", position=1_000_010, length=1_005.0, training_support=100),
    ]
    first_run = cluster_sv_events(records)
    second_run = cluster_sv_events(records)
    assert first_run == second_run
    # Tie on support: highest support count is shared by indices 1 and 2;
    # tiebreak picks the smaller original index.
    assert 1 in first_run and sorted(first_run[1]) == [0, 1, 2]


def test_tandem_duplications_at_same_start_with_different_lengths() -> None:
    # Two tandem duplications anchored at the same start but with very
    # different lengths fail reciprocal overlap and stay separate.
    records = [
        VariantRecord(
            variant_id="DUP_short",
            variant_class=VariantClass.DUPLICATION_LONG,
            chromosome="1",
            position=2_000_000,
            length=1_000.0,
            allele_frequency=0.01,
            training_support=50,
        ),
        VariantRecord(
            variant_id="DUP_long",
            variant_class=VariantClass.DUPLICATION_LONG,
            chromosome="1",
            position=2_000_000,
            length=10_000.0,
            allele_frequency=0.01,
            training_support=50,
        ),
    ]
    cluster_map = cluster_sv_events(records)
    assert cluster_map == {0: [0], 1: [1]}

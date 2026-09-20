"""Cross-source SV matching: the GATK-SV clustering rules and the sequence-resolved sites."""
from __future__ import annotations

import numpy as np

from sv_pgs.sv_fusion import SvSites, candidate_pairs, sequence_resolved_sites


def _sites(rows: list[tuple[str, int, int, int, str]], duplications_are_insertions: bool) -> SvSites:
    chromosomes, starts, ends, sizes, kinds = zip(*rows)
    return SvSites(
        chromosomes=np.asarray(chromosomes),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        sizes=np.asarray(sizes, dtype=np.int64),
        kinds=np.asarray(kinds),
        duplications_are_insertions=duplications_are_insertions,
    )


# First source: a sequence-resolved panel (DUP = inserted copy, a point).
_PANEL = _sites(
    [
        ("chr1", 10_000, 12_000, 2_000, "DEL"),  # 0: overlaps gatksv 0 by 1600/2000
        ("chr1", 20_000, 21_000, 1_000, "DEL"),  # 1: gatksv 1 is 3x longer (size ratio fails)
        ("chr1", 30_050, 30_051, 800, "INS"),  # 2: 50 bp from gatksv 2 (INS)
        ("chr1", 40_000, 40_001, 1_500, "INS"),  # 3: at the END of gatksv 3 (DUP interval)
        ("chr1", 50_000, 50_001, 600, "DUP"),  # 4: sequence-resolved DUP vs gatksv 4 DUP interval start
        ("chr1", 60_000, 61_000, 1_000, "DEL"),  # 5: gatksv 5 is an INS (incompatible kinds)
        ("chr2", 10_000, 12_000, 2_000, "DEL"),  # 6: other chromosome than gatksv 0
        ("chr1", 70_200, 70_201, 900, "INS"),  # 7: 200 bp from gatksv 6 (too far)
    ],
    duplications_are_insertions=True,
)
# Second source: GATK-SV (DUP = the duplicated interval).
_GATKSV = _sites(
    [
        ("chr1", 10_400, 12_400, 2_000, "DEL"),  # 0
        ("chr1", 20_000, 23_000, 3_000, "DEL"),  # 1
        ("chr1", 30_000, 30_001, 900, "INS"),  # 2
        ("chr1", 38_500, 40_020, 1_520, "DUP"),  # 3
        ("chr1", 49_990, 50_590, 600, "DUP"),  # 4
        ("chr1", 60_000, 60_001, 1_000, "INS"),  # 5
        ("chr1", 70_000, 70_001, 900, "INS"),  # 6
    ],
    duplications_are_insertions=False,
)


def test_candidate_pairs_follow_the_gatksv_clustering_rules() -> None:
    pairs = candidate_pairs(_PANEL, _GATKSV)

    assert list(zip(pairs.first_rows.tolist(), pairs.second_rows.tolist())) == [(0, 0), (2, 2), (3, 3), (4, 4)]
    np.testing.assert_allclose(pairs.reciprocal_overlaps[0], 0.8)
    assert np.isnan(pairs.reciprocal_overlaps[1:]).all()
    assert pairs.breakpoint_distances.tolist() == [-1, 50, 20, 10]
    np.testing.assert_allclose(pairs.size_ratios, [1.0, 800 / 900, 1500 / 1520, 1.0])


def test_duplication_geometry_depends_on_the_source_representation() -> None:
    # Two GATK-SV-style DUP intervals are compared by reciprocal overlap.
    first = _sites([("chr1", 1_000, 3_000, 2_000, "DUP")], duplications_are_insertions=False)
    second = _sites([("chr1", 1_500, 3_500, 2_000, "DUP")], duplications_are_insertions=False)

    pairs = candidate_pairs(first, second)

    np.testing.assert_allclose(pairs.reciprocal_overlaps, [0.75])
    assert pairs.breakpoint_distances.tolist() == [-1]


# ---------------------------------------------------------------------------
# Two-source calibration, on simulations with a known genotype
# ---------------------------------------------------------------------------


def test_sequence_resolved_sites_start_after_the_shared_prefix() -> None:
    deleted = "ACGTTGCA" * 25
    inserted = "TTAGGC" * 20
    sites = sequence_resolved_sites(
        np.array(["chr1", "chr1", "chr1"]),
        np.array([1_000, 5_000, 9_000], dtype=np.int64),
        ["G" + deleted, "C", "T" + "A" * 120 + "G"],
        ["G", "C" + inserted, "T" + "C" * 90 + "G"],
    )

    assert sites.kinds.tolist() == ["DEL", "INS", "CPX"]
    assert sites.starts.tolist() == [1_001, 5_001, 9_001]
    assert sites.ends.tolist() == [1_201, 5_002, 9_121]
    assert sites.sizes.tolist() == [200, 120, 120]
    assert sites.duplications_are_insertions
    # The deletion lines up with the same event as GATK-SV writes it (POS 1000,
    # the padding base; END 1200, the last deleted base).
    gatksv = _sites([("chr1", 1_001, 1_201, 200, "DEL")], duplications_are_insertions=False)
    assert candidate_pairs(sites, gatksv).first_rows.tolist() == [0]

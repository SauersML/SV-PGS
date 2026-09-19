"""Converter pieces: core spans, value-matched background, TR loci, SV context."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.store_converter import (
    NO_DISTANCE,
    NO_LOCUS,
    NO_RECORD,
    core_spans,
    linear_recalibration,
    sv_context,
    tr_loci,
    value_matched_background,
)


def test_core_spans_cover_the_changed_reference_bases() -> None:
    starts, ends = core_spans(
        np.array([100, 200, 300, 400]),
        ["A", "GACGT", "C", "TAAAG"],
        ["G", "G", "CTTTT", "TCCG"],
    )

    # SNV: its base; deletion: the deleted bases after the anchor; insertion: the two bases
    # around the insertion point; complex: the differing core.
    assert starts.tolist() == [99, 200, 299, 400]
    assert ends.tolist() == [100, 204, 301, 403]


def test_value_matched_background_zeroes_only_predicted_backgrounds() -> None:
    dosage_milli = np.array(
        [
            [0, 1, 2, 3, 1000],  # single path (K = 1, m = 1): backgrounds 1 and 2
            [1, 2, 4, 5, 10],  # K = 10, m = 2: q = 2w / (1 + 10w), backgrounds 2 and 4
            [1, 2, 3, 4, 5],  # no kept path carries the ID: nothing is background
        ],
        dtype=np.uint16,
    )

    corrected = value_matched_background(dosage_milli, np.array([1, 10, 10]), np.array([1, 2, 0]))

    np.testing.assert_array_equal(
        corrected.dosage_milli,
        [[0, 0, 0, 3, 1000], [1, 0, 0, 5, 10], [1, 2, 3, 4, 5]],
    )
    assert corrected.dosage_milli.dtype == np.uint16
    assert corrected.zeroed.tolist() == [2, 2, 0]
    assert corrected.unmatched_low.tolist() == [1, 2, 5]


def test_value_matched_background_rejects_impossible_path_counts() -> None:
    dosage = np.zeros((1, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="carrying paths"):
        value_matched_background(dosage, np.array([2]), np.array([3]))
    with pytest.raises(ValueError, match="integer dosage thousandths"):
        value_matched_background(dosage.astype(np.float64), np.array([1]), np.array([1]))


def test_tr_loci_merge_intervals_a_record_bridges() -> None:
    intervals = (np.array([0, 20, 40, 100]), np.array([10, 30, 50, 110]))
    core_starts = np.array([5, 8, 45, 60, 95, 25])
    core_ends = np.array([6, 22, 46, 61, 105, 26])
    length_changes = np.array([0, -14, 3, 0, 5, 0])

    loci = tr_loci(*intervals, core_starts, core_ends, length_changes)

    # Record 1 spans intervals 0 and 1, so they are one locus.
    assert loci.record_locus.tolist() == [0, 0, 1, int(NO_LOCUS), 2, 0]
    assert loci.starts.tolist() == [0, 40, 100]
    assert loci.ends.tolist() == [30, 50, 110]
    assert loci.interval_counts.tolist() == [2, 1, 1]
    assert loci.record_counts.tolist() == [3, 1, 1]
    assert loci.length_changing_record_counts.tolist() == [1, 1, 1]


def test_tr_loci_need_sorted_disjoint_intervals() -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        tr_loci(np.array([0, 5]), np.array([10, 20]), np.array([1]), np.array([2]), np.array([0]))


def _naive_context(starts, ends, is_sv, is_common, bubbles, window):
    count = starts.shape[0]
    distance = np.full(count, int(NO_DISTANCE), dtype=np.int64)
    for record in range(count):
        others = [other for other in np.flatnonzero(is_common) if other != record]
        if others:
            distance[record] = min(
                max(0, starts[other] - ends[record], starts[record] - ends[other]) for other in others
            )
    hulls = {}
    for record in np.flatnonzero(is_sv):
        low, high = hulls.get(bubbles[record], (np.inf, -np.inf))
        hulls[bubbles[record]] = (min(low, starts[record]), max(high, ends[record]))
    nearby = np.array(
        [
            sum(
                1
                for bubble, (low, high) in hulls.items()
                if bubble != bubbles[record] and high > starts[record] - window and low < ends[record] + window
            )
            for record in range(count)
        ]
    )
    return distance, nearby


def test_sv_context_matches_a_brute_force_reference() -> None:
    generator = np.random.default_rng(3)
    count = 400
    starts = np.sort(generator.integers(0, 2_000_000, count))
    ends = starts + generator.integers(1, 8_000, count)
    is_sv = generator.random(count) < 0.4
    is_common = is_sv & (generator.random(count) < 0.3)
    # Records at one POS share a bubble; here, consecutive runs of 1-4 records.
    bubbles = np.repeat(np.arange(count), generator.integers(1, 5, count))[:count]

    context = sv_context(starts, ends, is_sv, is_common, bubbles, window=50_000)

    distance, nearby = _naive_context(starts, ends, is_sv, is_common, bubbles, 50_000)
    np.testing.assert_array_equal(context.nearest_common_sv_distance.astype(np.int64), distance)
    np.testing.assert_array_equal(context.sv_bubbles_nearby.astype(np.int64), nearby)
    # The pointer names a common SV other than the record, at that distance.
    for record in range(count):
        target = int(context.nearest_common_sv_record[record])
        assert target != record and is_common[target]
        assert max(0, starts[target] - ends[record], starts[record] - ends[target]) == distance[record]


def test_sv_context_without_common_svs_marks_every_record() -> None:
    starts = np.array([0, 100, 200])
    ends = starts + 10
    context = sv_context(starts, ends, np.array([True, False, True]), np.zeros(3, dtype=bool), np.array([0, 1, 2]))

    assert context.nearest_common_sv_distance.tolist() == [int(NO_DISTANCE)] * 3
    assert context.nearest_common_sv_record.tolist() == [int(NO_RECORD)] * 3
    assert context.sv_bubbles_nearby.tolist() == [1, 2, 1]


def test_linear_recalibration_keeps_group_means_and_scales_deviations() -> None:
    dosage_milli = np.array([[0, 1000, 2000, 0, 1000, 2000]], dtype=np.uint16)
    groups = np.array([0, 0, 0, 1, 1, 1])

    recalibrated = linear_recalibration(dosage_milli, groups, np.array([[0.5, 1.0]]))

    np.testing.assert_array_equal(recalibrated.dosage_milli, [[500, 1000, 1500, 0, 1000, 2000]])
    assert recalibrated.clipped.tolist() == [0]


def test_linear_recalibration_clips_and_counts_and_needs_every_kappa() -> None:
    dosage_milli = np.array([[0, 0, 0, 2000]], dtype=np.uint16)
    recalibrated = linear_recalibration(dosage_milli, np.zeros(4, dtype=np.int64), np.array([[1.5]]))
    # Mean 500; 500 + 1.5 (2000 - 500) = 2750 clips to 2000; 500 + 1.5 (0 - 500) = -250 clips to 0.
    np.testing.assert_array_equal(recalibrated.dosage_milli, [[0, 0, 0, 2000]])
    assert recalibrated.clipped.tolist() == [4]
    with pytest.raises(ValueError, match="positive finite kappa"):
        linear_recalibration(dosage_milli, np.zeros(4, dtype=np.int64), np.array([[np.nan]]))

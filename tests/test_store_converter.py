"""Converter pieces: core spans, value-matched background, TR loci, SV context."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.dosage_store import (
    CodeArray,
    DosageStore,
    VariantTable,
    dosage_array_directory,
    encode_dosage_milli,
    sites_md5,
    write_manifest,
    write_variant_columns,
)
from sv_pgs.store_converter import (
    NO_DISTANCE,
    NO_LOCUS,
    NO_RECORD,
    ExpectedSites,
    assemble_half,
    core_spans,
    decode_batch,
    linear_recalibration,
    refalt_digest,
    sv_context,
    tr_loci,
    unbreakable_group_first,
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


def test_unbreakable_group_first_merges_overlapping_group_spans() -> None:
    bubbles = np.array([0, 0, -1, 1, -1, 1, -1, -1, 2, -1])
    tr_loci_ids = np.array([-1, -1, -1, -1, -1, 7, 7, 7, -1, -1])

    group_first = unbreakable_group_first(bubbles, tr_loci_ids)

    # Bubble 0 spans rows 0-1; bubble 1 spans 3-5 and TR locus 7 spans 5-7, so 3-7 is one span.
    assert group_first.tolist() == [0, 0, 2, 3, 3, 3, 3, 3, 8, 9]


_SITES = [
    # (POS, REF, ALT, ID, kept paths, carrying paths)
    (1_000, "A", "G", "snv1", 1, 1),
    (2_000, "C", "CTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT", "ins1", 10, 2),
    (3_000, "G", "T", "snv2", 1, 0),
]


def _write_batch(path, dosages) -> None:
    """A popped batch: GT:DS:GP with GP consistent with DS; ``dosages`` is [records][samples]."""
    samples = len(dosages[0])
    lines = [
        "##fileformat=VCFv4.2",
        '##INFO=<ID=ID,Number=1,Type=String,Description="atomic id">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">',
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="dosage">',
        '##FORMAT=<ID=GP,Number=G,Type=Float,Description="genotype probabilities">',
        "##contig=<ID=chr22,length=50818468>",
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + [f"s{index}" for index in range(samples)]),
    ]
    for (position, ref, alt, identifier, _, _), row in zip(_SITES, dosages):
        fields = []
        for dosage in row:
            second = max(dosage - 1.0, 0.0)
            first = dosage - 2.0 * second
            fields.append(f"0|0:{dosage:.3f}:{1.0 - first - second:.3f},{first:.3f},{second:.3f}")
        lines.append("\t".join(["chr22", str(position), ".", ref, alt, ".", "PASS", f"ID={identifier}", "GT:DS:GP", *fields]))
    path.write_text("\n".join(lines) + "\n")


def _expected_sites(positions_override=None) -> ExpectedSites:
    positions = np.array([site[0] for site in _SITES], dtype=np.int64)
    if positions_override is not None:
        positions = positions_override
    return ExpectedSites(
        positions=positions,
        refalt_digests=np.array([refalt_digest(site[1], site[2]) for site in _SITES], dtype=np.uint64),
        identifiers=tuple(site[3] for site in _SITES),
        kept_paths=np.array([site[4] for site in _SITES]),
        carrying_paths=np.array([site[5] for site in _SITES]),
    )


_BATCH_DOSAGES = (
    [[0.001, 1.0, 0.002], [0.002, 0.004, 1.5], [0.001, 0.0, 2.0]],
    [[0.002, 0.5], [1.0, 0.004], [0.003, 0.001]],
)


def test_decode_and_assemble_write_background_corrected_codes(tmp_path) -> None:
    expected = _expected_sites()
    batches = []
    for index, dosages in enumerate(_BATCH_DOSAGES):
        path = tmp_path / f"batch{index}.vcf"
        _write_batch(path, dosages)
        groups = np.zeros(len(dosages[0]), dtype=np.int64)
        batches.append(decode_batch(path, expected, groups, 1, tmp_path / f"batch{index}.npy"))

    root = tmp_path / "store"
    sums, squares = assemble_half(root, 0, "chr22", batches, np.zeros(5, dtype=np.int64), None, codec="raw")

    dosage_milli = np.rint(np.hstack([np.asarray(dosages) for dosages in _BATCH_DOSAGES]) * 1000).astype(np.uint16)
    corrected = value_matched_background(dosage_milli, expected.kept_paths, expected.carrying_paths).dosage_milli
    expected_codes = encode_dosage_milli(corrected)
    # Single-path snv1 loses its 1 and 2 milli backgrounds; ins1 (K = 10, m = 2) its 2 and 4;
    # snv2's paths were never kept (m = 0), so its 1 and 3 milli stay.
    assert corrected.tolist() == [[0, 1000, 0, 0, 500], [0, 0, 1500, 1000, 0], [1, 0, 2000, 3, 1]]
    table = VariantTable(
        chromosome=np.full(3, 22, dtype=np.int8),
        position=expected.positions,
        genetic_position_cm=np.zeros(3),
        ref_length=np.array([len(site[1]) for site in _SITES], dtype=np.int32),
        alt_length=np.array([len(site[2]) for site in _SITES], dtype=np.int32),
        variant_class=np.zeros(3, dtype=np.uint8),
        group_first=np.arange(3, dtype=np.int64),
        sum_code=sums.astype(np.uint64),
        sum_code2=squares.astype(np.uint64),
        annotations={},
        annotation_legends={},
        id_bytes=np.frombuffer("".join(expected.identifiers).encode(), dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum([len(identifier) for identifier in expected.identifiers])]).astype(np.int64),
    )
    write_variant_columns(root, "chr22", table, slice(0, 3))
    write_manifest(
        root,
        chromosomes=["chr22"],
        record_counts=[3],
        half_sample_counts=[5],
        chromosome_sites_md5=[sites_md5(table.position, table.ref_length, table.alt_length)],
    )
    store = DosageStore.open(root)
    np.testing.assert_array_equal(store.read_codes(0, 3), expected_codes)
    assert batches[0].zeroed.tolist() == [2, 2, 0] and batches[1].zeroed.tolist() == [1, 1, 0]
    assert batches[0].group_sums[:, 0].tolist() == [1000, 1500, 2001]


def test_decode_batch_fails_closed_on_a_sidecar_mismatch(tmp_path) -> None:
    path = tmp_path / "batch.vcf"
    _write_batch(path, _BATCH_DOSAGES[1])
    shifted = _expected_sites(np.array([1_000, 2_001, 3_000], dtype=np.int64))

    with pytest.raises(ValueError, match="record 1: POS differs"):
        decode_batch(path, shifted, np.zeros(2, dtype=np.int64), 1, tmp_path / "codes.npy")


def test_assemble_half_applies_the_linear_recalibration(tmp_path) -> None:
    expected = _expected_sites()
    path = tmp_path / "batch.vcf"
    _write_batch(path, _BATCH_DOSAGES[0])
    batch = decode_batch(path, expected, np.array([0, 0, 1]), 2, tmp_path / "codes.npy")
    scales = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])

    assemble_half(tmp_path / "store", 0, "chr22", [batch], np.array([0, 0, 1]), scales, codec="raw")

    dosage_milli = ((batch.codes.astype(np.int64) * 2000 + 127) // 254).astype(np.uint16)
    recalibrated = encode_dosage_milli(linear_recalibration(dosage_milli, np.array([0, 0, 1]), scales).dosage_milli)
    array = CodeArray(dosage_array_directory(tmp_path / "store", 0, "chr22"))
    target = np.empty((3, 3), dtype=np.uint8)
    array.read_rows_into(0, 3, target)
    np.testing.assert_array_equal(target, recalibrated)

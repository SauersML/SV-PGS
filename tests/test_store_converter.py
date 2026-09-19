"""Converter pieces: core spans, value-matched background, TR loci, SV context."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.dosage_store import (
    MISSING_CODE,
    CodeArray,
    HalfSamples,
    DosageStore,
    VariantTable,
    dosage_array_directory,
    encode_dosage_milli,
    open_column,
    read_manifest,
    sites_md5,
    statistic_column_directory,
    write_manifest,
    write_variant_columns,
)
from sv_pgs.store_converter import (
    NO_LOCUS,
    ExpectedSites,
    assemble_half,
    core_spans,
    decode_batch,
    decode_called_batch,
    linear_recalibration,
    no_call_fill,
    refalt_digest,
    sv_kernel_features,
    tr_loci,
    unbreakable_group_first,
    value_matched_background,
    write_store_manifest,
)


def _budget(host_bytes: int) -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu",
        device_ids=(),
        device_names=(),
        device_bytes=(),
        device_compute_capabilities=(),
        host_bytes=host_bytes,
        cpu_threads=1,
    )


_BUDGET = _budget(1 << 30)
# One byte of budget leaves room for one row per step.
_ONE_ROW_BUDGET = _budget(1)


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


def _naive_kernel(starts, ends, bubbles, is_sv, frequencies, classes, lengths, class_count, spacing):
    """Pair by pair, with scipy's cubic B-spline design matrix on the same uniform knots."""
    extent = int(ends.max() - starts.min())
    basis_count = int(np.floor(np.log1p(extent) / spacing)) + 4
    knots = spacing * np.arange(-3, basis_count + 1, dtype=np.float64)
    overlap = np.zeros((starts.size, class_count))
    distance = np.zeros((starts.size, class_count, basis_count, 2))
    for record in range(starts.size):
        for allele in np.flatnonzero(is_sv):
            if bubbles[allele] == bubbles[record]:
                continue
            weight = 2 * frequencies[allele] * (1 - frequencies[allele])
            if starts[allele] < ends[record] and starts[record] < ends[allele]:
                overlap[record, classes[allele]] += weight
                continue
            gap = max(0, starts[allele] - ends[record], starts[record] - ends[allele])
            basis = BSpline.design_matrix(np.array([np.log1p(gap)]), knots, 3).toarray()[0]
            distance[record, classes[allele], :, 0] += weight * basis
            distance[record, classes[allele], :, 1] += weight * np.log(lengths[allele]) * basis
    return overlap, distance


def _kernel_scenario(generator):
    count = 120
    starts = np.sort(generator.integers(0, 3_000_000, count))
    ends = starts + generator.integers(1, 20_000, count)
    bubbles = np.repeat(np.arange(count), generator.integers(1, 4, count))[:count]
    is_sv = generator.random(count) < 0.4
    frequencies = generator.uniform(0.0, 1.0, count)
    classes = generator.integers(0, 3, count)
    lengths = generator.integers(50, 20_000, count).astype(np.float64)
    return starts, ends, bubbles, is_sv, frequencies, classes, lengths


def test_sv_kernel_features_match_a_pairwise_bspline_reference() -> None:
    starts, ends, bubbles, is_sv, frequencies, classes, lengths = _kernel_scenario(np.random.default_rng(3))
    spacing = np.log(2.0)

    features = sv_kernel_features(starts, ends, bubbles, is_sv, frequencies, classes, lengths, 3, spacing, _BUDGET)
    one_row = sv_kernel_features(starts, ends, bubbles, is_sv, frequencies, classes, lengths, 3, spacing, _ONE_ROW_BUDGET)

    overlap, distance = _naive_kernel(starts, ends, bubbles, is_sv, frequencies, classes, lengths, 3, spacing)
    # Both sides sum the same terms in different orders: each term rounds a few times at most,
    # and a record sums at most one term per SV allele per basis function.
    eps = np.finfo(np.float64).eps
    scale = int(is_sv.sum()) * 16 * eps * max(1.0, float(np.log(lengths.max())))
    np.testing.assert_allclose(features.overlap, overlap, rtol=0.0, atol=scale)
    np.testing.assert_allclose(features.distance, distance, rtol=0.0, atol=scale)
    np.testing.assert_array_equal(one_row.overlap, features.overlap)
    np.testing.assert_allclose(one_row.distance, features.distance, rtol=0.0, atol=scale)
    assert features.distance[..., 0].sum() > 0 and features.overlap.sum() > 0


def test_sv_kernel_features_skip_the_own_bubble_and_separate_nesting() -> None:
    # Record 0 is an SV; record 1 sits inside it; record 2 is 10 bases past its end; record 3
    # is another allele of record 0's bubble.
    starts = np.array([100, 150, 510, 100])
    ends = np.array([500, 151, 511, 300])
    bubbles = np.array([0, 1, 2, 0])
    is_sv = np.array([True, False, False, True])
    frequencies = np.array([0.5, 0.0, 0.0, 0.1])
    classes = np.array([1, 0, 0, 0])
    lengths = np.array([400.0, 1.0, 1.0, 200.0])

    features = sv_kernel_features(starts, ends, bubbles, is_sv, frequencies, classes, lengths, 2, np.log(2.0), _BUDGET)

    # Record 0 and record 3 share a bubble, so neither counts the other.
    assert features.overlap[0].tolist() == [0.0, 0.0] and features.distance[0].sum() == 0.0
    assert features.overlap[3].tolist() == [0.0, 0.0] and features.distance[3].sum() == 0.0
    # Record 1 is nested in both alleles: weights 2 * 0.1 * 0.9 (class 0) and 2 * 0.5 * 0.5 (class 1).
    np.testing.assert_allclose(features.overlap[1], [0.18, 0.5], rtol=4 * np.finfo(np.float64).eps)
    assert features.distance[1].sum() == 0.0
    # Record 2: gap 10 to record 0 and 210 to record 3; each allele's basis weights sum to 1
    # (partition of unity), times its weight, and times its log length in the second slot.
    np.testing.assert_allclose(features.distance[2, 1, :, 0].sum(), 0.5, rtol=8 * np.finfo(np.float64).eps)
    np.testing.assert_allclose(features.distance[2, 0, :, 0].sum(), 0.18, rtol=8 * np.finfo(np.float64).eps)
    np.testing.assert_allclose(features.distance[2, 0, :, 1].sum(), 0.18 * np.log(200.0), rtol=8 * np.finfo(np.float64).eps)


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


def _reported_info(identifier: str) -> float:
    """The synthetic INFO/INFO of a site."""
    return {"snv1": 0.98, "ins1": 0.61, "snv2": 0.3}[identifier]


def _write_batch(path, dosages, omit_info=False) -> None:
    """A popped batch: GT:DS:GP with GP consistent with DS, and INFO/INFO; ``dosages`` is [records][samples]."""
    samples = len(dosages[0])
    lines = [
        "##fileformat=VCFv4.2",
        '##INFO=<ID=ID,Number=1,Type=String,Description="atomic id">',
        '##INFO=<ID=INFO,Number=1,Type=Float,Description="imputation r2">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">',
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="dosage">',
        '##FORMAT=<ID=GP,Number=G,Type=Float,Description="genotype probabilities">',
        "##contig=<ID=chr22,length=50818468>",
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + [f"{path.stem}_s{index}" for index in range(samples)]),
    ]
    for (position, ref, alt, identifier, _, _), row in zip(_SITES, dosages):
        fields = []
        for dosage in row:
            second = max(dosage - 1.0, 0.0)
            first = dosage - 2.0 * second
            fields.append(f"0|0:{dosage:.3f}:{1.0 - first - second:.3f},{first:.3f},{second:.3f}")
        info = "" if omit_info else f";INFO={_reported_info(identifier):.3f}"
        lines.append("\t".join(["chr22", str(position), ".", ref, alt, ".", "PASS", f"ID={identifier}{info}", "GT:DS:GP", *fields]))
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
        batches.append(decode_batch(path, expected, groups, 1, tmp_path / f"batch{index}.npy", _BUDGET))

    root = tmp_path / "store"
    sums, squares = assemble_half(root, 0, "chr22", [batch.codes for batch in batches], np.zeros(5, dtype=np.int64), None, _fill_of(batches), codec="raw", budget=_BUDGET)

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
    # Budget-sized steps change how many rows each step handles, never the result.
    for index, dosages in enumerate(_BATCH_DOSAGES):
        one_row = decode_batch(
            tmp_path / f"batch{index}.vcf", expected, np.zeros(len(dosages[0]), dtype=np.int64), 1, tmp_path / f"row{index}.npy", _ONE_ROW_BUDGET
        )
        np.testing.assert_array_equal(one_row.codes, batches[index].codes)
        np.testing.assert_array_equal(one_row.group_sums, batches[index].group_sums)
    one_row_sums, one_row_squares = assemble_half(
        tmp_path / "one_row_store", 0, "chr22", [batch.codes for batch in batches], np.zeros(5, dtype=np.int64), None,
        _fill_of(batches), codec="raw", budget=_ONE_ROW_BUDGET,
    )
    np.testing.assert_array_equal(one_row_sums, sums)
    np.testing.assert_array_equal(one_row_squares, squares)
    assert batches[0].group_sums[:, 0].tolist() == [1000, 1500, 2001]


def test_decode_batch_fails_closed_on_a_sidecar_mismatch(tmp_path) -> None:
    path = tmp_path / "batch.vcf"
    _write_batch(path, _BATCH_DOSAGES[1])
    shifted = _expected_sites(np.array([1_000, 2_001, 3_000], dtype=np.int64))

    with pytest.raises(ValueError, match="record 1: POS differs"):
        decode_batch(path, shifted, np.zeros(2, dtype=np.int64), 1, tmp_path / "codes.npy", _BUDGET)


def test_decode_batch_returns_info_from_the_same_read_and_needs_it(tmp_path) -> None:
    path = tmp_path / "batch.vcf"
    _write_batch(path, _BATCH_DOSAGES[1])
    decoded = decode_batch(path, _expected_sites(), np.zeros(2, dtype=np.int64), 1, tmp_path / "codes.npy", _BUDGET)
    # htslib holds INFO floats as float32.
    np.testing.assert_allclose(decoded.reported_info, [_reported_info(site[3]) for site in _SITES], rtol=np.finfo(np.float32).eps)

    _write_batch(path, _BATCH_DOSAGES[1], omit_info=True)
    with pytest.raises(ValueError, match="record 0: INFO/INFO missing"):
        decode_batch(path, _expected_sites(), np.zeros(2, dtype=np.int64), 1, tmp_path / "codes.npy", _BUDGET)


def test_assemble_half_applies_the_linear_recalibration(tmp_path) -> None:
    expected = _expected_sites()
    path = tmp_path / "batch.vcf"
    _write_batch(path, _BATCH_DOSAGES[0])
    batch = decode_batch(path, expected, np.array([0, 0, 1]), 2, tmp_path / "codes.npy", _BUDGET)
    scales = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])

    assemble_half(tmp_path / "store", 0, "chr22", [batch.codes], np.array([0, 0, 1]), scales, _fill_of([batch]), codec="raw", budget=_BUDGET)

    dosage_milli = ((batch.codes.astype(np.int64) * 2000 + 127) // 254).astype(np.uint16)
    recalibrated = encode_dosage_milli(linear_recalibration(dosage_milli, np.array([0, 0, 1]), scales).dosage_milli)
    array = CodeArray(dosage_array_directory(tmp_path / "store", 0, "chr22"))
    target = np.empty((3, 3), dtype=np.uint8)
    array.read_rows_into(0, 3, target)
    np.testing.assert_array_equal(target, recalibrated)


def _fill_of(batches):
    return no_call_fill(sum(batch.group_sums for batch in batches), sum(batch.group_counts for batch in batches))


def _write_called_batch(path, genotypes) -> None:
    """Hard calls only (GT), as the long-read panel members' genotypes come; ``genotypes`` [records][samples]."""
    samples = len(genotypes[0])
    lines = [
        "##fileformat=VCFv4.2",
        '##INFO=<ID=ID,Number=1,Type=String,Description="atomic id">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">',
        "##contig=<ID=chr22,length=50818468>",
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + [f"{path.stem}_lr{index}" for index in range(samples)]),
    ]
    for (position, ref, alt, identifier, _, _), row in zip(_SITES, genotypes):
        lines.append("\t".join(["chr22", str(position), ".", ref, alt, ".", "PASS", f"ID={identifier}", "GT", *row]))
    path.write_text("\n".join(lines) + "\n")


def test_a_long_read_half_joins_the_imputed_halves_on_the_same_sites(tmp_path) -> None:
    expected = _expected_sites()
    imputed = []
    for index, dosages in enumerate(_BATCH_DOSAGES):
        path = tmp_path / f"batch{index}.vcf"
        _write_batch(path, dosages)
        imputed.append(decode_batch(path, expected, np.zeros(len(dosages[0]), dtype=np.int64), 1, tmp_path / f"imputed{index}.npy", _BUDGET))
    called_path = tmp_path / "long_read.vcf"
    _write_called_batch(called_path, [["0|1", "1|1"], ["0|0", "1|0"], ["1|1", "0|0"]])
    long_read = decode_called_batch(called_path, expected, np.zeros(2, dtype=np.int64), 1, tmp_path / "long_read.npy", _BUDGET)

    root = tmp_path / "store"
    fill = _fill_of([*imputed, long_read])
    imputed_sums, imputed_squares = assemble_half(root, 0, "chr22", [batch.codes for batch in imputed], np.zeros(5, dtype=np.int64), None, fill, codec="raw", budget=_BUDGET)
    long_read_sums, long_read_squares = assemble_half(root, 1, "chr22", [long_read.codes], np.zeros(2, dtype=np.int64), None, fill, codec="zstd", budget=_BUDGET)
    positions = expected.positions
    ref_lengths = np.array([len(site[1]) for site in _SITES], dtype=np.int32)
    alt_lengths = np.array([len(site[2]) for site in _SITES], dtype=np.int32)
    table = VariantTable(
        chromosome=np.full(3, 22, dtype=np.int8),
        position=positions,
        genetic_position_cm=np.zeros(3),
        ref_length=ref_lengths,
        alt_length=alt_lengths,
        variant_class=np.zeros(3, dtype=np.uint8),
        group_first=np.arange(3, dtype=np.int64),
        sum_code=(imputed_sums + long_read_sums).astype(np.uint64),
        sum_code2=(imputed_squares + long_read_squares).astype(np.uint64),
        annotations={},
        annotation_legends={},
        id_bytes=np.frombuffer("".join(expected.identifiers).encode(), dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum([len(identifier) for identifier in expected.identifiers])]).astype(np.int64),
    )
    write_variant_columns(root, "chr22", table, slice(0, 3))
    write_store_manifest(
        root,
        chromosomes=["chr22"],
        record_counts=[3],
        chromosome_sites_md5=[sites_md5(positions, ref_lengths, alt_lengths)],
        half_sample_counts=[5, 2],
        half_samples=[
            HalfSamples("dragen_sample", tuple(name for batch in imputed for name in batch.sample_ids)),
            HalfSamples("research_id", long_read.sample_ids),
        ],
        half_measurements=["imputed_dosage", "long_read_calls"],
        gates={"S0": "PASS"},
        recalibrated=False,
    )

    store = DosageStore.open(root)
    assert store.half_samples() == (
        HalfSamples("dragen_sample", (*imputed[0].sample_ids, *imputed[1].sample_ids)),
        HalfSamples("research_id", long_read.sample_ids),
    )
    codes = store.read_codes(0, 3)
    assert codes.shape == (3, 7)
    np.testing.assert_array_equal(codes[:, 5:], [[127, 254], [0, 127], [254, 0]])
    assert long_read.zeroed.tolist() == [0, 0, 0] and long_read.reported_info.size == 0
    assert store.statistic("no_calls").tolist() == [0, 0, 0]
    assert read_manifest(root)["attributes"]["half_measurements"] == ["imputed_dosage", "long_read_calls"]


def test_the_store_manifest_refuses_a_sample_listed_twice_in_a_half(tmp_path) -> None:
    manifest = dict(
        chromosomes=["chr22"],
        record_counts=[3],
        chromosome_sites_md5=["0" * 32],
        half_sample_counts=[2],
        half_measurements=["imputed_dosage"],
        gates={},
        recalibrated=False,
    )
    with pytest.raises(ValueError, match="more than once"):
        write_store_manifest(tmp_path / "twice", half_samples=[HalfSamples("dragen_sample", ("s1", "s1"))], **manifest)
    with pytest.raises(ValueError, match="one sample name per sample"):
        write_store_manifest(tmp_path / "short", half_samples=[HalfSamples("dragen_sample", ("s1",))], **manifest)


def test_a_long_read_no_call_takes_its_groups_measured_mean(tmp_path) -> None:
    expected = _expected_sites()
    first, second = tmp_path / "first.vcf", tmp_path / "second.vcf"
    _write_called_batch(first, [["0|1", "./."], ["0|0", "1|1"], ["1|1", "0|0"]])
    _write_called_batch(second, [["1|1", "./."], ["./.", "0|1"], ["1|1", "0|1"]])
    batches = [
        decode_called_batch(path, expected, np.array([0, 1]), 2, tmp_path / f"{path.stem}.npy", _BUDGET)
        for path in (first, second)
    ]
    assert batches[0].codes[0].tolist() == [127, MISSING_CODE]
    assert batches[0].no_calls.tolist() == [1, 0, 0]

    root = tmp_path / "store"
    assemble_half(root, 0, "chr22", [batch.codes for batch in batches], np.array([0, 1, 0, 1]), None, _fill_of(batches), codec="raw", budget=_BUDGET)

    # Record 0: group 1 has no measurement, so both take the pooled mean (1000 + 2000) / 2.
    # Record 1: sample 2's group 0 measured 0|0 in the other batch, so 0.
    codes = np.empty((3, 4), dtype=np.uint8)
    CodeArray(dosage_array_directory(root, 0, "chr22")).read_rows_into(0, 3, codes)
    np.testing.assert_array_equal(codes, [[127, 191, 254, 191], [0, 254, 0, 127], [254, 0, 254, 127]])
    assert open_column(statistic_column_directory(root, 0, "chr22", "no_calls"))[0].tolist() == [2, 1, 0]


def test_a_record_no_sample_measured_fails(tmp_path) -> None:
    path = tmp_path / "long_read.vcf"
    _write_called_batch(path, [["0|1", "1|1"], ["./.", "./."], ["1|1", "0|0"]])
    batch = decode_called_batch(path, _expected_sites(), np.zeros(2, dtype=np.int64), 1, tmp_path / "codes.npy", _BUDGET)

    with pytest.raises(ValueError, match="record 1: no sample has a measurement"):
        _fill_of([batch])

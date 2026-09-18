from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.dosage_store import (
    FLOAT32_EXACT_ROWS,
    INT32_EXACT_ROWS,
    MAXIMUM_CODE,
    CodeArray,
    CodeShardWriter,
    DosageStore,
    QuantizedDosageMatrix,
    create_code_array,
    decode_codes,
    encode_dosage_milli,
    exact_signed_gram,
    open_column,
    signed_code_moments,
    signed_codes,
    statistic_column_directory,
    variant_column_directory,
    write_column,
    write_manifest,
    write_variant_ids,
    dosage_array_directory,
)

SHARD_ROWS = 128
INNER_ROWS = 16


def _random_dosage_milli(rng: np.random.Generator, rows: int, samples: int) -> np.ndarray:
    """3-decimal ALT dosages with the imputed-data mix: calls, err-imp floors and diffuse values."""
    calls = rng.choice(np.array([2, 1000, 1998, 0, 2000], dtype=np.int64), size=(rows, samples))
    diffuse = rng.integers(0, 2001, size=(rows, samples))
    return np.where(rng.random((rows, samples)) < 0.3, diffuse, calls)


def _write_store(root: Path, milli_by_half: list[dict[str, np.ndarray]]) -> None:
    chromosomes = list(milli_by_half[0])
    record_counts = [milli_by_half[0][chromosome].shape[0] for chromosome in chromosomes]
    write_manifest(
        root,
        chromosomes=chromosomes,
        record_counts=record_counts,
        half_sample_counts=[half[chromosomes[0]].shape[1] for half in milli_by_half],
    )
    for chromosome, record_count in zip(chromosomes, record_counts):
        positions = np.arange(1, record_count + 1, dtype=np.int32) * 100
        write_column(variant_column_directory(root, chromosome, "pos"), positions)
        write_column(variant_column_directory(root, chromosome, "ref_len"), np.ones(record_count, dtype=np.int32))
        write_column(variant_column_directory(root, chromosome, "alt_len"), np.ones(record_count, dtype=np.int32))
        write_column(
            variant_column_directory(root, chromosome, "class"),
            np.zeros(record_count, dtype=np.uint8),
            {"legend": ["SNV", "INDEL", "SV"]},
        )
        write_variant_ids(root, chromosome, [f"{chromosome}-{position}-allele0-1" for position in positions])
    for half_index, half in enumerate(milli_by_half):
        for chromosome, milli in half.items():
            codes = encode_dosage_milli(milli)
            directory = dosage_array_directory(root, half_index, chromosome)
            layout = create_code_array(directory, codes.shape[0], codes.shape[1], shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS)
            for shard_index in range(layout.shard_count):
                with CodeShardWriter(directory, layout, shard_index) as writer:
                    shard_rows = codes[shard_index * SHARD_ROWS : (shard_index + 1) * SHARD_ROWS]
                    for start in range(0, shard_rows.shape[0], 37):
                        writer.write_rows(shard_rows[start : start + 37])
            wide = codes.astype(np.int64)
            write_column(statistic_column_directory(root, half_index, chromosome, "sum_code"), wide.sum(axis=1).astype(np.uint64))
            write_column(
                statistic_column_directory(root, half_index, chromosome, "sum_code2"), (wide * wide).sum(axis=1).astype(np.uint64)
            )


@pytest.fixture()
def two_half_store(tmp_path: Path) -> tuple[Path, list[dict[str, np.ndarray]]]:
    rng = np.random.default_rng(20260918)
    milli_by_half = [
        {"chr21": _random_dosage_milli(rng, 300, 37), "chr22": _random_dosage_milli(rng, 150, 37)},
        {"chr21": _random_dosage_milli(rng, 300, 23), "chr22": _random_dosage_milli(rng, 150, 23)},
    ]
    _write_store(tmp_path / "store", milli_by_half)
    return tmp_path / "store", milli_by_half


def _all_codes(milli_by_half: list[dict[str, np.ndarray]]) -> np.ndarray:
    return np.hstack([np.vstack([encode_dosage_milli(milli) for milli in half.values()]) for half in milli_by_half])


def test_every_milli_dosage_maps_to_its_nearest_code_rounding_half_up() -> None:
    milli = np.arange(2001)
    codes = encode_dosage_milli(milli)
    for value, code in zip(milli.tolist(), codes.tolist()):
        exact = Fraction(value * 127, 1000)
        assert code == int(exact + Fraction(1, 2))
        assert abs(Fraction(code, 127) - Fraction(value, 1000)) <= Fraction(1, 254)
    assert codes[[0, 1000, 2000]].tolist() == [0, 127, 254]
    assert decode_codes(codes[[0, 1000, 2000]]).tolist() == [0.0, 1.0, 2.0]
    assert np.all(np.diff(codes.astype(int)) >= 0)
    assert set(codes.tolist()) == set(range(MAXIMUM_CODE + 1))


def test_encoding_rejects_non_integer_and_out_of_range_dosages() -> None:
    with pytest.raises(TypeError):
        encode_dosage_milli(np.array([0.5]))
    with pytest.raises(ValueError):
        encode_dosage_milli(np.array([2001]))
    with pytest.raises(ValueError):
        decode_codes(np.array([255], dtype=np.uint8))


def test_signed_codes_shift_every_code_exactly_in_every_output_dtype() -> None:
    codes = np.arange(MAXIMUM_CODE + 1, dtype=np.uint8)
    expected = np.arange(MAXIMUM_CODE + 1) - 127
    for dtype in (np.int8, np.int16, np.int32, np.float32, np.float64):
        assert np.array_equal(signed_codes(codes, np.empty(codes.shape, dtype=dtype)), expected)


def test_store_round_trip_matches_quantized_float_dosage(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, milli_by_half = two_half_store
    expected_codes = _all_codes(milli_by_half)
    float_dosage = np.hstack([np.vstack(list(half.values())) for half in milli_by_half]) / 1000.0
    with DosageStore(root) as store:
        assert (store.variant_count, store.sample_count) == expected_codes.shape
        out = np.empty((store.variant_count, store.sample_count), dtype=np.uint8)
        read_back = store.read_codes_into(0, store.variant_count, out)
        assert np.array_equal(read_back, expected_codes)
        assert np.max(np.abs(decode_codes(read_back) - float_dosage)) <= 1 / 254 + 1e-12
        rng = np.random.default_rng(3)
        for _ in range(40):
            start = int(rng.integers(0, store.variant_count))
            stop = int(rng.integers(start, store.variant_count + 1))
            assert np.array_equal(store.codes(start, stop, out), expected_codes[start:stop])
        blocks = [(start, min(store.variant_count, start + 70)) for start in range(0, store.variant_count, 70)]
        buffers = [np.empty((70, store.sample_count), dtype=np.uint8) for _ in range(3)]
        for start, stop, block in store.iter_code_blocks(blocks, buffers):
            assert np.array_equal(block, expected_codes[start:stop])
        for start, stop, block in store.iter_code_views(blocks, buffers[0]):
            assert np.array_equal(block, expected_codes[start:stop])
        assert store.variants.variant_ids([0, 300, 449]) == ["chr21-100-allele0-1", "chr22-100-allele0-1", "chr22-15000-allele0-1"]
        assert store.variants.legends["class"] == ("SNV", "INDEL", "SV")
        assert store.variants.chromosome_indices(np.array([0, 299, 300, 449])).tolist() == [0, 0, 1, 1]


def test_single_half_ranges_inside_a_shard_are_zero_copy_views(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, milli_by_half = two_half_store
    half_codes = np.vstack([encode_dosage_milli(milli) for milli in milli_by_half[1].values()])
    with DosageStore(root, half_indices=[1]) as store:
        spare = np.zeros((SHARD_ROWS, store.sample_count), dtype=np.uint8)
        view = store.codes(10, 100, spare)
        assert not view.flags.writeable and not view.flags.owndata
        assert not np.shares_memory(view, spare)
        assert np.array_equal(view, half_codes[10:100])
        crossing = store.codes(100, 200, spare)
        assert np.shares_memory(crossing, spare)
        assert np.array_equal(crossing, half_codes[100:200])


def test_corrupted_shard_index_and_missing_code_fail_loudly(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, _ = two_half_store
    shard_path = dosage_array_directory(root, 0, "chr21") / "c" / "0" / "0"
    payload = bytearray(shard_path.read_bytes())
    payload[-6] ^= 0xFF
    shard_path.write_bytes(bytes(payload))
    array = CodeArray(dosage_array_directory(root, 0, "chr21"))
    with pytest.raises(ValueError, match="crc32c"):
        array.read_rows_into(0, 4, np.empty((4, 37), dtype=np.uint8))
    layout = create_code_array(root / "bad", 4, 3, shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS)
    with pytest.raises(ValueError, match="never stored"):
        with CodeShardWriter(root / "bad", layout, 0) as writer:
            writer.write_rows(np.full((4, 3), 255, dtype=np.uint8))
    assert not (root / "bad" / "c" / "0" / "0.partial").exists()


def test_code_array_metadata_is_zarr_v3_sharded(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, _ = two_half_store
    metadata = json.loads((dosage_array_directory(root, 0, "chr22") / "zarr.json").read_text())
    assert metadata["zarr_format"] == 3 and metadata["shape"] == [150, 37]
    sharding = metadata["codecs"][0]
    assert sharding["name"] == "sharding_indexed"
    assert sharding["configuration"]["chunk_shape"] == [INNER_ROWS, 37]
    shard_bytes = (dosage_array_directory(root, 0, "chr22") / "c" / "1" / "0").read_bytes()
    index_entries = SHARD_ROWS // INNER_ROWS
    index = np.frombuffer(shard_bytes[-(16 * index_entries + 4) : -4], dtype="<u8").reshape(index_entries, 2)
    written = -(-(150 - SHARD_ROWS) // INNER_ROWS)
    assert index[:written, 0].tolist() == [chunk * INNER_ROWS * 37 for chunk in range(written)]
    assert np.all(index[written:] == np.uint64(2**64 - 1))


def test_training_moments_are_exact_and_standardize_like_float_dosage(
    two_half_store: tuple[Path, list[dict[str, np.ndarray]]],
) -> None:
    root, milli_by_half = two_half_store
    codes = _all_codes(milli_by_half)
    rng = np.random.default_rng(11)
    for training_fraction in (0.3, 0.8):
        training_rows = np.flatnonzero(rng.random(codes.shape[1]) < training_fraction)
        with DosageStore(root) as store:
            moments = signed_code_moments(store, 20, 420, training_rows, block_rows=64)
            matrix = QuantizedDosageMatrix(store, 20, 420, training_rows, moments)
            standardized = matrix.standardized_block(20, 420)
        signed = codes[20:420][:, training_rows].astype(object) - 127
        assert moments.signed_sums.tolist() == [int(value) for value in signed.sum(axis=1)]
        assert moments.signed_square_sums.tolist() == [int(value) for value in (signed * signed).sum(axis=1)]
        dosage = decode_codes(codes[20:420][:, training_rows])
        reference = (dosage - dosage.mean(axis=1, keepdims=True)) / dosage.std(axis=1, keepdims=True)
        assert np.max(np.abs(standardized - reference)) < 1e-12
        assert np.allclose(moments.dosage_means, dosage.mean(axis=1), rtol=0, atol=1e-14)
        assert np.allclose(moments.dosage_scales, dosage.std(axis=1), rtol=1e-13, atol=0)


def test_folded_products_match_the_dense_standardized_design(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, milli_by_half = two_half_store
    codes = _all_codes(milli_by_half)
    rng = np.random.default_rng(5)
    training_rows = np.flatnonzero(rng.random(codes.shape[1]) < 0.7)
    held_out_rows = np.setdiff1d(np.arange(codes.shape[1]), training_rows)
    with DosageStore(root) as store:
        training = QuantizedDosageMatrix.from_training_rows(store, 0, 450, training_rows, block_rows=50)
        held_out = QuantizedDosageMatrix(store, 0, 450, held_out_rows, training.moments)
        coefficients = rng.standard_normal(450)
        vector = rng.standard_normal(training_rows.size)
        for matrix in (training, held_out):
            dense = matrix.standardized_block(0, 450).T
            assert np.allclose(matrix.matvec(coefficients, block_rows=64), dense @ coefficients, rtol=0, atol=1e-10)
            assert np.allclose(matrix.gram(280, 330), dense[:, 280:330].T @ dense[:, 280:330], rtol=0, atol=1e-9)
        dense_training = training.standardized_block(0, 450).T
        assert np.allclose(training.transpose_matvec(vector, block_rows=64), dense_training.T @ vector, rtol=0, atol=1e-10)
        assert np.allclose(np.diag(training.gram(0, 100)), training_rows.size, rtol=1e-12)


def test_sidecar_sums_that_disagree_with_the_codes_raise(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, _ = two_half_store
    sums, _ = open_column(statistic_column_directory(root, 1, "chr22", "sum_code"), writable=True)
    sums[7] += 1
    sums.flush()
    with DosageStore(root) as store:
        with pytest.raises(ValueError, match="sidecar"):
            signed_code_moments(store, 0, store.variant_count, np.arange(store.sample_count), block_rows=64)


def test_zero_training_variance_is_refused_not_floored(tmp_path: Path) -> None:
    milli = _random_dosage_milli(np.random.default_rng(1), 20, 9)
    milli[4] = 2
    _write_store(tmp_path / "store", [{"chr1": milli}])
    with DosageStore(tmp_path / "store") as store:
        with pytest.raises(ValueError, match="zero training variance"):
            QuantizedDosageMatrix.from_training_rows(store, 0, 20, np.arange(9), block_rows=8)


def test_int32_accumulator_bound_is_exact_and_tight() -> None:
    assert INT32_EXACT_ROWS == 133_144
    worst_product = np.int64(127 * 127)
    exact_total = worst_product * INT32_EXACT_ROWS
    assert exact_total <= np.iinfo(np.int32).max
    products = np.full(INT32_EXACT_ROWS + 1, worst_product, dtype=np.int32)
    running = np.add.accumulate(products, dtype=np.int32)
    assert int(running[INT32_EXACT_ROWS - 1]) == int(exact_total)
    assert int(running[INT32_EXACT_ROWS]) != int(exact_total + worst_product)
    mixed = np.where(np.arange(INT32_EXACT_ROWS) % 2 == 0, 127, -127).astype(np.int8)
    assert int(np.add.reduce(mixed.astype(np.int32) * mixed.astype(np.int32), dtype=np.int32)) == int(exact_total)


def test_float32_accumulator_bound_is_exact_and_tight() -> None:
    assert FLOAT32_EXACT_ROWS == 1_040
    products = np.full(FLOAT32_EXACT_ROWS + 1, 127.0 * 127.0, dtype=np.float32)
    running = np.add.accumulate(products, dtype=np.float32)
    assert float(running[FLOAT32_EXACT_ROWS - 1]) == 127 * 127 * FLOAT32_EXACT_ROWS
    assert float(running[FLOAT32_EXACT_ROWS]) != 127 * 127 * (FLOAT32_EXACT_ROWS + 1)


def test_exact_signed_gram_is_exact_on_worst_case_codes_past_both_bounds() -> None:
    rng = np.random.default_rng(9)
    row_count = 3 * FLOAT32_EXACT_ROWS + 17
    signed = rng.choice(np.array([-127, 127, -126, 1], dtype=np.int16), size=(6, row_count))
    signed[0] = 127
    signed[1] = -127
    reference = np.array(
        [[sum(int(left) * int(right) for left, right in zip(row_a, row_b)) for row_b in signed] for row_a in signed]
    )
    assert np.array_equal(exact_signed_gram(signed), reference)

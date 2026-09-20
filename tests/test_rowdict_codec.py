from pathlib import Path

import numpy as np
import pytest

from sv_pgs import rowdict_codec
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.dosage_store import CodeArray, CodeShardWriter, DosageStore, create_code_array, transcode_store
from tests.test_dosage_store import _all_codes, _two_half_dosage, _write_store


def _rows_of_every_depth(rng: np.random.Generator, samples: int) -> np.ndarray:
    """Rows whose best depth is each k in 0..8: 2^k equally common values plus a few rare ones."""
    rows = []
    for depth in range(rowdict_codec.MAXIMUM_DEPTH + 1):
        values = rng.choice(255, size=min(1 << depth, 255), replace=False).astype(np.uint8)
        row = values[rng.integers(0, values.shape[0], size=samples)]
        if depth < rowdict_codec.MAXIMUM_DEPTH:
            rare = rng.choice(samples, size=min(3, samples), replace=False)
            row[rare] = rng.choice(np.setdiff1d(np.arange(255), values), size=rare.shape[0]).astype(np.uint8)
        rows.append(row)
    rows.append(np.full(samples, 254, dtype=np.uint8))
    rows.append(rng.integers(0, 255, size=samples).astype(np.uint8))
    return np.vstack(rows)


def _decode_chunk(payload: bytes, chunk_rows: int, samples: int) -> np.ndarray:
    # The store's chain appends a crc32c after the payload; a decoder may read into it.
    buffer = np.frombuffer(payload + bytes(4), dtype=np.uint8)
    frames = rowdict_codec.row_frames(
        buffer, np.array([0]), np.array([len(payload)]), chunk_rows, samples, np.arange(chunk_rows, dtype=np.int64)
    )
    out = np.full((chunk_rows, samples), 255, dtype=np.uint8)
    rowdict_codec.decode_rows(buffer, frames, samples, out)
    return out


def _frame_sizes(payload: bytes, chunk_rows: int) -> np.ndarray:
    return np.frombuffer(payload[: chunk_rows * rowdict_codec.ROW_SIZE_DTYPE.itemsize], dtype=rowdict_codec.ROW_SIZE_DTYPE)


@pytest.mark.parametrize("samples", [1, 7, 4099])
def test_chunks_round_trip_exactly_at_every_depth(samples: int) -> None:
    rng = np.random.default_rng(samples)
    codes = _rows_of_every_depth(rng, samples)
    payload = rowdict_codec.encode_chunk(codes)
    assert np.array_equal(_decode_chunk(payload, codes.shape[0], samples), codes)
    if samples == 4099:
        sizes = _frame_sizes(payload, codes.shape[0]).astype(np.int64)
        starts = codes.shape[0] * 4 + np.cumsum(sizes) - sizes
        depths = np.frombuffer(payload, dtype=np.uint8)[starts]
        assert depths[: rowdict_codec.MAXIMUM_DEPTH + 1].tolist() == list(range(rowdict_codec.MAXIMUM_DEPTH + 1))


def test_each_frame_is_the_smallest_over_every_depth() -> None:
    rng = np.random.default_rng(11)
    samples = 1000
    codes = np.vstack([_rows_of_every_depth(rng, samples), rng.binomial(254, rng.uniform(0, 0.02, (20, 1)), (20, samples)).astype(np.uint8)])
    sizes = _frame_sizes(rowdict_codec.encode_chunk(codes), codes.shape[0])
    for row, size in zip(codes, sizes.tolist()):
        counts = np.sort(np.bincount(row, minlength=256))[::-1]
        by_depth = [
            1 + (1 << depth) + -(-samples * depth // 8) + 3 * int(samples - counts[: 1 << depth].sum())
            for depth in range(9)
        ]
        assert size == min(by_depth)


def test_wide_rows_store_exception_samples_as_uint32() -> None:
    samples = (1 << 16) + 3
    rng = np.random.default_rng(5)
    codes = np.zeros((3, samples), dtype=np.uint8)
    codes[0, [0, 65535, 65536, samples - 1]] = [9, 17, 33, 254]
    codes[1] = rng.integers(0, 4, size=samples)
    codes[1, [65537, samples - 1]] = [200, 201]
    codes[2] = rng.integers(0, 255, size=samples)
    assert rowdict_codec.exception_sample_dtype(samples) == np.dtype("<u4")
    assert rowdict_codec.exception_sample_dtype(1 << 16) == np.dtype("<u2")
    payload = rowdict_codec.encode_chunk(codes)
    assert _frame_sizes(payload, 3)[0] == 1 + 1 + 0 + 4 * (4 + 1)
    assert np.array_equal(_decode_chunk(payload, 3, samples), codes)


def test_a_size_table_that_disagrees_with_the_chunk_is_refused() -> None:
    codes = np.random.default_rng(2).integers(0, 3, size=(4, 50)).astype(np.uint8)
    payload = bytearray(rowdict_codec.encode_chunk(codes))
    payload[0] += 1
    with pytest.raises(ValueError, match="size table"):
        _decode_chunk(bytes(payload), 4, 50)
    payload[0] -= 1
    payload[4] += 1
    payload[8] -= 1
    with pytest.raises(ValueError, match="row frame"):
        _decode_chunk(bytes(payload), 4, 50)


def _cpu_budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(),
        host_bytes=1 << 30, cpu_threads=4,
    )


def test_a_rowdict_store_reads_every_range_exactly(tmp_path: Path) -> None:
    milli_by_half = _two_half_dosage()
    _write_store(tmp_path / "zstd", milli_by_half, "zstd")
    transcode_store(tmp_path / "zstd", tmp_path / "rowdict", codec="rowdict", budget=_cpu_budget())
    expected = _all_codes(milli_by_half)
    rng = np.random.default_rng(8)
    with DosageStore.open(tmp_path / "rowdict") as store:
        assert np.array_equal(store.read_codes(0, store.n_variants), expected)
        for _ in range(30):
            start = int(rng.integers(0, store.n_variants))
            stop = int(rng.integers(start, store.n_variants + 1))
            columns = np.sort(rng.choice(store.n_samples, size=11, replace=False))
            assert np.array_equal(store.read_codes(start, stop), expected[start:stop])
            assert np.array_equal(store.read_codes(start, stop, columns), expected[start:stop][:, columns])


def test_the_last_partial_chunk_of_a_shard_is_padded_and_read_exactly(tmp_path: Path) -> None:
    codes = np.random.default_rng(4).integers(0, 255, size=(21, 13)).astype(np.uint8)
    layout = create_code_array(tmp_path, 21, 13, codec="rowdict", shard_rows=16, inner_rows=8)
    for shard_index in range(layout.shard_count):
        with CodeShardWriter(tmp_path, layout, shard_index) as writer:
            writer.write_rows(codes[shard_index * 16 : (shard_index + 1) * 16])
    array = CodeArray(tmp_path)
    out = np.empty((21, 13), dtype=np.uint8)
    array.read_rows_into(0, 21, out)
    assert np.array_equal(out, codes)
    wide = np.zeros((4, 20), dtype=np.uint8)
    array.read_rows_into(15, 19, wide[:, 3:16])
    assert np.array_equal(wide[:, 3:16], codes[15:19]) and not wide[:, :3].any() and not wide[:, 16:].any()
    array.close()


def test_the_crc32c_table_and_combine_match_google_crc32c() -> None:
    import google_crc32c

    table = rowdict_codec._crc32c_table()
    rng = np.random.default_rng(31)
    for length in (0, 1, 7, 1000, 4097):
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
        crc = 0xFFFFFFFF
        for byte in data:
            crc = int(table[(crc ^ byte) & 0xFF]) ^ (crc >> 8)
        assert crc ^ 0xFFFFFFFF == google_crc32c.value(data)
        for cut in (0, length // 3, length):
            first, second = data[:cut], data[cut:]
            combined = rowdict_codec._multiply_mod_polynomial(rowdict_codec.byte_shift(len(second)), google_crc32c.value(first))
            assert combined ^ google_crc32c.value(second) == google_crc32c.value(data)


def test_the_crc32c_table_and_combine_match_google_crc32c() -> None:
    import google_crc32c

    table = rowdict_codec._crc32c_table()
    rng = np.random.default_rng(31)
    for length in (0, 1, 7, 1000, 4097):
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
        crc = 0xFFFFFFFF
        for byte in data:
            crc = int(table[(crc ^ byte) & 0xFF]) ^ (crc >> 8)
        assert crc ^ 0xFFFFFFFF == google_crc32c.value(data)
        for cut in (0, length // 3, length):
            first, second = data[:cut], data[cut:]
            combined = rowdict_codec._multiply_mod_polynomial(rowdict_codec.byte_shift(len(second)), google_crc32c.value(first))
            assert combined ^ google_crc32c.value(second) == google_crc32c.value(data)

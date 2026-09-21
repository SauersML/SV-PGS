from pathlib import Path

import numpy as np
import pytest

from sv_pgs import dosage_store, rowdict_codec
from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.dosage_store import CodeArray, CodeShardWriter, DosageStore, create_code_array, transcode_store
from tests.test_dosage_store import _all_codes, _two_half_dosage, _write_store
from tests.test_rowdict_codec import _cpu_budget, _rows_of_every_depth

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


@pytest.mark.parametrize("samples", [1, 7, 4099, (1 << 16) + 5])
def test_the_gpu_decoder_matches_the_cpu_reference_at_every_depth(tmp_path: Path, samples: int) -> None:
    rng = np.random.default_rng(samples)
    codes = np.vstack([_rows_of_every_depth(rng, samples) for _ in range(3)])
    layout = create_code_array(tmp_path, codes.shape[0], samples, codec="rowdict", shard_rows=16, inner_rows=4)
    for shard_index in range(layout.shard_count):
        with CodeShardWriter(tmp_path, layout, shard_index) as writer:
            writer.write_rows(codes[shard_index * 16 : (shard_index + 1) * 16])
    array = CodeArray(tmp_path)
    decoder = rowdict_codec.GpuRowDecoder(cupy)
    for start, stop in ((0, codes.shape[0]), (3, 19), (5, 6), (17, codes.shape[0])):
        reference = np.empty((stop - start, samples), dtype=np.uint8)
        array.read_rows_into(start, stop, reference)
        on_device = cupy.full((stop - start, samples + 9), 255, dtype=cupy.uint8)
        array.read_rows_to_device(start, stop, on_device[:, 4 : 4 + samples], decoder)
        decoded = cupy.asnumpy(on_device)
        assert np.array_equal(decoded[:, 4 : 4 + samples], reference)
        assert np.array_equal(reference, codes[start:stop])
        assert np.all(decoded[:, :4] == 255) and np.all(decoded[:, 4 + samples :] == 255)
    array.close()


def test_a_two_half_rowdict_store_decodes_on_the_device(tmp_path: Path) -> None:
    milli_by_half = _two_half_dosage()
    _write_store(tmp_path / "zstd", milli_by_half, "zstd")
    transcode_store(tmp_path / "zstd", tmp_path / "rowdict", codec="rowdict", budget=_cpu_budget())
    expected = _all_codes(milli_by_half)
    decoder = rowdict_codec.GpuRowDecoder(cupy)
    with DosageStore.open(tmp_path / "rowdict") as store:
        for start, stop in ((0, store.n_variants), (290, 310), (0, 1), (449, 450)):
            out = cupy.empty((stop - start, store.n_samples), dtype=cupy.uint8)
            assert np.array_equal(cupy.asnumpy(store.read_codes_to_device(start, stop, out, decoder)), expected[start:stop])
    with DosageStore.open(tmp_path / "zstd") as store, pytest.raises(ValueError, match="rowdict"):
        store.read_codes_to_device(0, 1, cupy.empty((1, store.n_samples), dtype=cupy.uint8), decoder)


def _rowdict_array(directory: Path, codes: np.ndarray, shard_rows: int, inner_rows: int) -> CodeArray:
    layout = create_code_array(directory, codes.shape[0], codes.shape[1], codec="rowdict", shard_rows=shard_rows, inner_rows=inner_rows)
    for shard_index in range(layout.shard_count):
        with CodeShardWriter(directory, layout, shard_index) as writer:
            writer.write_rows(codes[shard_index * shard_rows : (shard_index + 1) * shard_rows])
    return CodeArray(directory)


def test_selected_rows_decode_into_a_compact_target_reading_only_their_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(31)
    samples = 301
    codes = np.vstack([_rows_of_every_depth(rng, samples) for _ in range(9)])
    array = _rowdict_array(tmp_path, codes, shard_rows=32, inner_rows=4)
    decoder = rowdict_codec.GpuRowDecoder(cupy)
    windows: list[int] = []
    acquire = dosage_store._PINNED_POOL.acquire
    monkeypatch.setattr(dosage_store._PINNED_POOL, "acquire", lambda module, count: windows.append(count) or acquire(module, count))
    read: list[tuple[int, int]] = []
    pread = dosage_store._pread_exact
    monkeypatch.setattr(dosage_store, "_pread_exact", lambda fd, buffers, offset: read.append((offset, sum(len(b) for b in buffers))) or pread(fd, buffers, offset))
    everything = np.arange(codes.shape[0])
    selections = [
        everything,
        everything[::4],  # one row per chunk: every chunk is read, a quarter of the frames decoded
        np.array([0, 1, 2, 70, 71, codes.shape[0] - 1]),  # whole chunks and shards skipped
        np.array([33]),
        rng.choice(codes.shape[0], size=40, replace=False),
    ]
    for rows in selections:
        rows = np.sort(rows)
        windows.clear()
        read.clear()
        on_device = cupy.full((rows.shape[0], samples + 6), 255, dtype=cupy.uint8)
        # A non-blocking stream is not ordered after the fill on the default stream.
        cupy.cuda.Device().synchronize()
        with cupy.cuda.Stream(non_blocking=True) as stream:
            array.read_rows_to_device(0, codes.shape[0], on_device[:, 3 : 3 + samples], decoder, rows)
            stream.synchronize()
        decoded = cupy.asnumpy(on_device)
        assert np.array_equal(decoded[:, 3 : 3 + samples], codes[rows])
        assert np.all(decoded[:, :3] == 255) and np.all(decoded[:, 3 + samples :] == 255)
        chunks = np.unique(rows // 4)
        chunk_bytes = [
            int(array._shard(int(chunk) * 4 // 32).chunk_sizes[int(chunk) % 8]) for chunk in chunks
        ]
        # Exactly the wanted chunks are read, and the staging window never exceeds the decoded rows
        # it serves (or one chunk, when one chunk is larger).
        assert sum(size for _, size in read) == sum(chunk_bytes)
        assert windows == [max(max(chunk_bytes), min(sum(chunk_bytes), rows.shape[0] * samples))]
    array.close()


def test_a_selective_read_through_many_windows_matches_the_codes(tmp_path: Path) -> None:
    rng = np.random.default_rng(32)
    samples = 97
    # Uniform codes make every frame dense (k = 8), so one row per chunk needs several windows.
    codes = rng.integers(0, 255, size=(96, samples)).astype(np.uint8)
    array = _rowdict_array(tmp_path, codes, shard_rows=96, inner_rows=8)
    decoder = rowdict_codec.GpuRowDecoder(cupy)
    rows = np.arange(3, 96, 8)
    out = cupy.empty((rows.shape[0], samples), dtype=cupy.uint8)
    array.read_rows_to_device(0, 96, out, decoder, rows)
    assert np.array_equal(cupy.asnumpy(out), codes[rows])
    array.close()


def test_store_selective_reads_span_chromosomes_and_halves(tmp_path: Path) -> None:
    milli_by_half = _two_half_dosage()
    _write_store(tmp_path / "zstd", milli_by_half, "zstd")
    transcode_store(tmp_path / "zstd", tmp_path / "rowdict", codec="rowdict", budget=_cpu_budget())
    expected = _all_codes(milli_by_half)
    decoder = rowdict_codec.GpuRowDecoder(cupy)
    rng = np.random.default_rng(33)
    with DosageStore.open(tmp_path / "rowdict") as store:
        for start, stop, count in ((0, store.n_variants, 57), (280, 320, 9), (299, 301, 2), (10, 20, 0)):
            rows = np.sort(rng.choice(np.arange(start, stop), size=count, replace=False))
            out = cupy.empty((count, store.n_samples), dtype=cupy.uint8)
            assert np.array_equal(cupy.asnumpy(store.read_codes_to_device(start, stop, out, decoder, rows)), expected[rows])
        with pytest.raises(ValueError, match="ascending"):
            store.read_codes_to_device(0, 10, cupy.empty((2, store.n_samples), dtype=cupy.uint8), decoder, np.array([5, 3]))
        with pytest.raises(ValueError, match="ascending"):
            store.read_codes_to_device(0, 10, cupy.empty((1, store.n_samples), dtype=cupy.uint8), decoder, np.array([10]))


def _array_of_every_depth(tmp_path: Path, samples: int) -> tuple[CodeArray, np.ndarray]:
    rng = np.random.default_rng(samples + 3)
    codes = np.vstack([_rows_of_every_depth(rng, samples) for _ in range(2)])
    layout = create_code_array(tmp_path, codes.shape[0], samples, codec="rowdict", shard_rows=16, inner_rows=4)
    for shard_index in range(layout.shard_count):
        with CodeShardWriter(tmp_path, layout, shard_index) as writer:
            writer.write_rows(codes[shard_index * 16 : (shard_index + 1) * 16])
    return CodeArray(tmp_path), codes


def test_the_device_checks_refuse_a_corrupted_chunk(tmp_path: Path) -> None:
    import google_crc32c

    array, codes = _array_of_every_depth(tmp_path, 4099)
    decoder = rowdict_codec.GpuRowDecoder(cupy)
    shard = array._shard(0)
    offset, size = int(shard.chunk_offsets[1]), int(shard.chunk_sizes[1]) - 4
    array.close()
    path = tmp_path / "c" / "0" / "0"
    original = bytearray(path.read_bytes())

    def read(start: int, stop: int) -> None:
        fresh = CodeArray(tmp_path)
        try:
            fresh.read_rows_to_device(start, stop, cupy.empty((stop - start, 4099), dtype=cupy.uint8), decoder)
        finally:
            fresh.close()

    corrupted = bytearray(original)
    corrupted[offset + size // 2] ^= 0x10
    path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="inner chunk 1 fails its crc32c"):
        read(0, 16)
    read(0, 4)  # chunk 0 alone is intact
    # a size table that no longer covers the payload, under a matching crc32c
    corrupted = bytearray(original)
    corrupted[offset] ^= 0x01
    corrupted[offset + size : offset + size + 4] = google_crc32c.value(bytes(corrupted[offset : offset + size])).to_bytes(4, "little")
    path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="row size table"):
        read(4, 8)
    path.write_bytes(bytes(original))
    out = cupy.empty((16, 4099), dtype=cupy.uint8)
    fresh = CodeArray(tmp_path)
    fresh.read_rows_to_device(0, 16, out, decoder)
    fresh.close()
    assert np.array_equal(cupy.asnumpy(out), codes[:16])


def test_the_device_decode_runs_on_a_non_blocking_stream(tmp_path: Path) -> None:
    array, codes = _array_of_every_depth(tmp_path, (1 << 16) + 5)
    decoder = rowdict_codec.GpuRowDecoder(cupy)
    stream = cupy.cuda.Stream(non_blocking=True)
    out = cupy.zeros(codes.shape, dtype=cupy.uint8)
    with stream:
        array.read_rows_to_device(0, codes.shape[0], out, decoder)
    stream.synchronize()
    array.close()
    assert np.array_equal(cupy.asnumpy(out), codes)

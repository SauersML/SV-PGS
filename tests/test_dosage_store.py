from fractions import Fraction
import json
import os
import resource
from pathlib import Path
import signal
import time

import numpy as np
import pytest

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import VariantClass
from sv_pgs.dosage_store import (
    MANIFEST_FILE,
    MAXIMUM_CODE,
    VARIANT_CLASS_LEGEND,
    VARIANT_CLASSES,
    CodeArray,
    CodeShardWriter,
    DosageStore,
    StoreTileSource,
    VariantTable,
    create_code_array,
    dosage_array_directory,
    encode_dosage_milli,
    open_column,
    read_manifest,
    sites_md5,
    statistic_column_directory,
    transcode_store,
    variant_column_directory,
    write_column,
    write_dosage_store,
    write_manifest,
    write_variant_ids,
)
from sv_pgs.genotype_buffers import FLOAT32_EXACT_ROWS, INT32_EXACT_ROWS, SIGNED_CODE_OFFSET, HostGenotypeBuffer, build_sample_layout
from sv_pgs.genotype_statistics import plan_genotype_pass, run_genotype_pass
from tests.stage0_support import InMemoryTileSource

SHARD_ROWS = 128
INNER_ROWS = 16


def _budget(host_bytes: int = 1 << 30) -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu",
        device_ids=(),
        device_names=(),
        device_bytes=(),
        device_compute_capabilities=(),
        host_bytes=host_bytes,
        cpu_threads=4,
    )


def _random_dosage_milli(rng: np.random.Generator, rows: int, samples: int) -> np.ndarray:
    """3-decimal ALT dosages with the imputed-data mix: calls, err-imp floors and diffuse values."""
    calls = rng.choice(np.array([2, 1000, 1998, 0, 2000], dtype=np.int64), size=(rows, samples))
    diffuse = rng.integers(0, 2001, size=(rows, samples))
    return np.where(rng.random((rows, samples)) < 0.3, diffuse, calls)


def _write_store(root: Path, milli_by_half: list[dict[str, np.ndarray]], codec: str = "raw") -> None:
    chromosomes = list(milli_by_half[0])
    record_counts = [milli_by_half[0][chromosome].shape[0] for chromosome in chromosomes]
    digests = []
    for chromosome, record_count in zip(chromosomes, record_counts):
        positions = np.arange(1, record_count + 1, dtype=np.int64) * 100
        lengths = np.ones(record_count, dtype=np.int32)
        columns = {
            "pos": (positions, {}),
            "ref_len": (lengths, {}),
            "alt_len": (lengths, {}),
            "cm": (positions / 1e6, {}),
            "variant_class": (np.zeros(record_count, dtype=np.uint8), {"legend": VARIANT_CLASS_LEGEND}),
            "group_first": (np.arange(record_count, dtype=np.int64), {}),
            "class": (np.zeros(record_count, dtype=np.uint8), {"legend": ["SNV", "INDEL", "SV"]}),
            "has_pl": (np.arange(record_count) % 3 == 0, {}),
            "panel_af": (np.full(record_count, 0.25, dtype=np.float32), {}),
        }
        for name, (values, attributes) in columns.items():
            write_column(variant_column_directory(root, chromosome, name), values, attributes)
        write_variant_ids(root, chromosome, [f"{chromosome}-{position}-allele0-1" for position in positions])
        digests.append(sites_md5(positions, lengths, lengths))
    write_manifest(
        root,
        chromosomes=chromosomes,
        record_counts=record_counts,
        half_sample_counts=[half[chromosomes[0]].shape[1] for half in milli_by_half],
        chromosome_sites_md5=digests,
    )
    for half_index, half in enumerate(milli_by_half):
        for chromosome, milli in half.items():
            codes = encode_dosage_milli(milli)
            directory = dosage_array_directory(root, half_index, chromosome)
            layout = create_code_array(
                directory, codes.shape[0], codes.shape[1], codec=codec, shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS
            )
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


def _two_half_dosage() -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260918)
    return [
        {"chr21": _random_dosage_milli(rng, 300, 37), "chr22": _random_dosage_milli(rng, 150, 37)},
        {"chr21": _random_dosage_milli(rng, 300, 23), "chr22": _random_dosage_milli(rng, 150, 23)},
    ]


@pytest.fixture(params=["raw", "zstd", "rowdict"])
def two_half_store(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, list[dict[str, np.ndarray]]]:
    milli_by_half = _two_half_dosage()
    _write_store(tmp_path / "store", milli_by_half, request.param)
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
    assert (codes[[0, 1000, 2000]] / 127).tolist() == [0.0, 1.0, 2.0]
    assert np.all(np.diff(codes.astype(int)) >= 0)
    assert set(codes.tolist()) == set(range(MAXIMUM_CODE + 1))


def test_encoding_rejects_non_integer_and_out_of_range_dosages() -> None:
    with pytest.raises(TypeError):
        encode_dosage_milli(np.array([0.5]))
    with pytest.raises(ValueError):
        encode_dosage_milli(np.array([2001]))


def test_store_round_trip_matches_quantized_float_dosage(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, milli_by_half = two_half_store
    expected_codes = _all_codes(milli_by_half)
    float_dosage = np.hstack([np.vstack(list(half.values())) for half in milli_by_half]) / 1000.0
    rng = np.random.default_rng(3)
    with DosageStore.open(root) as store:
        assert (store.n_variants, store.n_samples) == expected_codes.shape
        read_back = store.read_codes(0, store.n_variants)
        assert read_back.flags.c_contiguous
        assert np.array_equal(read_back, expected_codes)
        # Quantization error 1/254, plus one rounding each of milli / 1000 and code / 127 (values
        # at most 2, so at most eps each) and of their difference.
        assert np.max(np.abs(read_back / 127 - float_dosage)) <= 1 / 254 + 3 * np.finfo(np.float64).eps
        gathered = np.sort(rng.choice(store.n_samples, size=25, replace=False))
        contiguous = np.arange(30, 50)
        for _ in range(40):
            start = int(rng.integers(0, store.n_variants))
            stop = int(rng.integers(start, store.n_variants + 1))
            assert np.array_equal(store.read_codes(start, stop), expected_codes[start:stop])
            for columns in (gathered, contiguous):
                assert np.array_equal(store.read_codes(start, stop, columns), expected_codes[start:stop][:, columns])
        blocks = [(start, min(store.n_variants, start + 70)) for start in range(0, store.n_variants, 70)]
        for budget in (_budget(), _budget(3 * 70 * store.n_samples)):
            for start, stop, block in store.iter_codes(blocks, gathered, budget):
                assert np.array_equal(block, expected_codes[start:stop][:, gathered])
            for start, stop, block in store.iter_codes(blocks, None, budget):
                assert np.array_equal(block, expected_codes[start:stop])
        with pytest.raises(MemoryError):
            next(store.iter_codes(blocks, None, _budget(70 * store.n_samples)))
        table = store.variant_table
        assert table.variant_ids([0, 300, 449]) == ["chr21-100-allele0-1", "chr22-100-allele0-1", "chr22-15000-allele0-1"]
        assert table.chromosome[[0, 299, 300, 449]].tolist() == [21, 21, 22, 22]
        assert table.annotation_legends == {"class": ("SNV", "INDEL", "SV"), "has_pl": ("false", "true")}
        assert table.annotations["has_pl"].dtype == np.int32 and table.annotations["panel_af"].dtype == np.float64
        assert np.array_equal(table.group_first, np.arange(450))
        assert np.array_equal(table.sum_code, expected_codes.astype(np.uint64).sum(axis=1))
    with DosageStore.open(root, half_indices=[1]) as half_store:
        assert np.array_equal(half_store.variant_table.sum_code, expected_codes[:, 37:].astype(np.uint64).sum(axis=1))


def test_raw_single_half_ranges_inside_a_shard_are_zero_copy_views(tmp_path: Path) -> None:
    milli_by_half = _two_half_dosage()
    _write_store(tmp_path / "raw", milli_by_half, "raw")
    _write_store(tmp_path / "zstd", milli_by_half, "zstd")
    half_codes = np.vstack([encode_dosage_milli(milli) for milli in milli_by_half[1].values()])
    with DosageStore.open(tmp_path / "raw", half_indices=[1]) as store:
        view = store.read_codes(10, 100)
        assert not view.flags.writeable and not view.flags.owndata and view.flags.c_contiguous
        assert np.array_equal(view, half_codes[10:100])
        assert store.read_codes(100, 200).flags.owndata
        views = [block for _, _, block in store.iter_codes([(0, 64), (64, 128), (120, 140)], None, _budget())]
        assert [block.flags.writeable for block in views[:2]] == [False, False]
        assert np.array_equal(views[1], half_codes[64:128])
    with DosageStore.open(tmp_path / "zstd", half_indices=[1]) as store:
        assert store.read_codes(10, 100).flags.owndata
        assert np.array_equal(store.read_codes(10, 100), half_codes[10:100])


def test_corrupted_shards_and_missing_codes_fail_loudly(tmp_path: Path) -> None:
    milli_by_half = _two_half_dosage()
    for codec, flipped_byte in (("raw", -6), ("zstd", 40), ("rowdict", 40)):
        root = tmp_path / codec
        _write_store(root, milli_by_half, codec)
        shard_path = dosage_array_directory(root, 0, "chr21") / "c" / "0" / "0"
        payload = bytearray(shard_path.read_bytes())
        payload[flipped_byte] ^= 0xFF
        shard_path.write_bytes(bytes(payload))
        array = CodeArray(dosage_array_directory(root, 0, "chr21"))
        with pytest.raises(ValueError, match="crc32c"):
            array.read_rows_into(0, 4, np.empty((4, 37), dtype=np.uint8))
    layout = create_code_array(tmp_path / "bad", 4, 3, codec="raw", shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS)
    with pytest.raises(ValueError, match="never stored"):
        with CodeShardWriter(tmp_path / "bad", layout, 0) as writer:
            writer.write_rows(np.full((4, 3), 255, dtype=np.uint8))
    assert not (tmp_path / "bad" / "c" / "0" / "0.partial").exists()


def test_open_checks_the_sites_md5(tmp_path: Path) -> None:
    _write_store(tmp_path / "store", _two_half_dosage())
    positions, _ = open_column(variant_column_directory(tmp_path / "store", "chr22", "pos"), writable=True)
    positions[5] += 1
    positions.flush()
    with pytest.raises(ValueError, match="md5"):
        DosageStore.open(tmp_path / "store")


def test_open_refuses_a_store_written_with_other_variant_classes(tmp_path: Path) -> None:
    _write_store(tmp_path / "store", _two_half_dosage())
    metadata_path = variant_column_directory(tmp_path / "store", "chr21", "variant_class") / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["attributes"]["legend"] = ["snv", "small_indel", "deletion_short", "deletion_long"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="variant classes"):
        DosageStore.open(tmp_path / "store")


def test_code_array_metadata_is_zarr_v3_sharded(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, _ = two_half_store
    metadata = json.loads((dosage_array_directory(root, 0, "chr22") / "zarr.json").read_text())
    assert metadata["zarr_format"] == 3 and metadata["shape"] == [150, 37]
    sharding = metadata["codecs"][0]
    assert sharding["name"] == "sharding_indexed"
    assert sharding["configuration"]["chunk_shape"] == [INNER_ROWS, 37]
    inner_codecs = [codec["name"] for codec in sharding["configuration"]["codecs"]]
    assert inner_codecs in (["bytes"], ["bytes", "zstd", "crc32c"], ["bytes", "svpgs_rowdict", "crc32c"])
    shard_bytes = (dosage_array_directory(root, 0, "chr22") / "c" / "1" / "0").read_bytes()
    index_entries = SHARD_ROWS // INNER_ROWS
    index = np.frombuffer(shard_bytes[-(16 * index_entries + 4) : -4], dtype="<u8").reshape(index_entries, 2)
    written = -(-(150 - SHARD_ROWS) // INNER_ROWS)
    assert index[0, 0] == 0 and np.all(index[1:written, 0] == np.cumsum(index[: written - 1, 1]))
    assert np.all(index[written:] == np.uint64(2**64 - 1))


@pytest.mark.parametrize("codec", ["raw", "zstd", "rowdict"])
def test_write_dosage_store_round_trips_table_and_codes(tmp_path: Path, codec: str) -> None:
    rng = np.random.default_rng(7)
    counts = (90, 41)
    codes = encode_dosage_milli(_random_dosage_milli(rng, sum(counts), 29))
    wide = codes.astype(np.uint64)
    chromosome = np.repeat(np.array([2, 5], dtype=np.int8), counts)
    variant_count = sum(counts)
    ids = b"".join(f"v{row}".encode() for row in range(variant_count))
    table = VariantTable(
        chromosome=chromosome,
        position=np.concatenate([np.arange(counts[0]) * 7 + 3, np.arange(counts[1]) * 5 + 1]).astype(np.int64),
        genetic_position_cm=np.linspace(0.0, 3.0, variant_count),
        ref_length=rng.integers(1, 4, variant_count).astype(np.int32),
        alt_length=rng.integers(1, 60, variant_count).astype(np.int32),
        variant_class=rng.integers(0, len(VARIANT_CLASSES), variant_count).astype(np.uint8),
        group_first=np.concatenate([np.arange(counts[0]) // 3 * 3, counts[0] + np.arange(counts[1]) // 2 * 2]).astype(np.int64),
        sum_code=wide.sum(axis=1),
        sum_code2=(wide * wide).sum(axis=1),
        annotations={"n_paths_total": rng.integers(1, 30, variant_count).astype(np.float64)},
        annotation_legends={},
        id_bytes=np.frombuffer(ids, dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum([len(f"v{row}") for row in range(variant_count)])]).astype(np.int64),
    )
    blocks = [codes[start : start + 17] for start in range(0, variant_count, 17)]
    write_dosage_store(tmp_path / "store", 29, table, blocks, codec=codec, shard_rows=32, inner_rows=8)
    with DosageStore.open(tmp_path / "store") as store:
        assert store.chromosomes == ("chr2", "chr5")
        assert np.array_equal(store.read_codes(0, variant_count), codes)
        read_table = store.variant_table
        for field_name in ("chromosome", "position", "ref_length", "alt_length", "variant_class", "group_first", "sum_code", "sum_code2"):
            assert np.array_equal(getattr(read_table, field_name), getattr(table, field_name)), field_name
        assert np.allclose(read_table.genetic_position_cm, table.genetic_position_cm)
        assert np.array_equal(read_table.annotations["n_paths_total"], table.annotations["n_paths_total"])
        assert read_table.variant_ids([0, 130]) == ["v0", "v130"]
        assert VARIANT_CLASSES[int(read_table.variant_class[0])] in set(VariantClass)
    manifest = json.loads((tmp_path / "store" / MANIFEST_FILE).read_text())
    assert manifest["half_sample_counts"] == [29]
    bad = VariantTable(**{**{name: getattr(table, name) for name in VariantTable.__slots__}, "sum_code": table.sum_code + 1})
    with pytest.raises(ValueError, match="sum_code"):
        write_dosage_store(tmp_path / "bad", 29, bad, blocks, codec=codec, shard_rows=32, inner_rows=8)


def test_transcoding_copies_the_whole_store_with_zero_copy_raw_halves(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, milli_by_half = two_half_store
    expected_codes = _all_codes(milli_by_half)
    cache_root = root.parent / "cache"
    # A budget of 50 rows per read cuts every shard into several pieces.
    transcode_store(root, cache_root, codec="raw", budget=_budget(50 * expected_codes.shape[1]))
    with DosageStore.open(root) as store, DosageStore.open(cache_root) as cache:
        assert cache.half_indices == store.half_indices and cache.n_samples == store.n_samples
        assert np.array_equal(cache.read_codes(0, cache.n_variants), expected_codes)
        assert np.array_equal(cache.variant_table.sum_code, store.variant_table.sum_code)
        assert cache.variant_table.annotation_legends == store.variant_table.annotation_legends
        assert cache.manifest_attributes == store.manifest_attributes
    assert read_manifest(cache_root) == read_manifest(root)
    for half in range(len(milli_by_half)):
        for chromosome in milli_by_half[half]:
            for name in ("sum_code", "sum_code2"):
                cached, _ = open_column(statistic_column_directory(cache_root, half, chromosome, name))
                stored, _ = open_column(statistic_column_directory(root, half, chromosome, name))
                assert np.array_equal(cached, stored)
    with DosageStore.open(cache_root, half_indices=(0,)) as first_half:
        view = first_half.read_codes(3, 90)
        assert not view.flags.owndata
        assert np.array_equal(view, expected_codes[3:90, : first_half.n_samples])


def test_opening_a_store_raises_the_soft_file_limit_to_the_hard_limit(
    two_half_store: tuple[Path, list[dict[str, np.ndarray]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A soft limit below the store's descriptors is raised to the hard limit, never to the
    # store's exact count, so later opens (a CUDA context, another store) still have room.
    root, _ = two_half_store
    requested: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (8, 4096))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, limits: requested.append(limits))
    with DosageStore.open(root):
        pass
    assert requested == [(4096, 4096)]


def test_a_store_without_a_sample_manifest_says_so(two_half_store: tuple[Path, list[dict[str, np.ndarray]]]) -> None:
    root, _ = two_half_store
    with DosageStore.open(root) as store:
        with pytest.raises(ValueError, match="no sample manifest"):
            store.half_samples()


def test_a_forked_child_compresses_with_its_own_threads(tmp_path: Path) -> None:
    layout = create_code_array(tmp_path / "parent", 40, 9, codec="zstd", shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS)
    codes = encode_dosage_milli(_random_dosage_milli(np.random.default_rng(4), 40, 9))
    with CodeShardWriter(tmp_path / "parent", layout, 0) as writer:
        writer.write_rows(codes)
    child = os.fork()
    if child == 0:
        exit_code = 1
        try:
            child_layout = create_code_array(tmp_path / "child", 40, 9, codec="zstd", shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS)
            with CodeShardWriter(tmp_path / "child", child_layout, 0) as writer:
                writer.write_rows(codes)
            exit_code = 0
        finally:
            os._exit(exit_code)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        finished, status = os.waitpid(child, os.WNOHANG)
        if finished:
            break
        time.sleep(0.05)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("the forked child hung on the parent's compression pool")
    assert os.waitstatus_to_exitcode(status) == 0
    read_back = np.empty((40, 9), dtype=np.uint8)
    CodeArray(tmp_path / "child").read_rows_into(0, 40, read_back)
    assert np.array_equal(read_back, codes)


def test_standardizing_signed_codes_equals_standardizing_the_decoded_dosage(
    two_half_store: tuple[Path, list[dict[str, np.ndarray]]],
) -> None:
    root, milli_by_half = two_half_store
    training = np.flatnonzero(np.random.default_rng(11).random(60) < 0.7)
    with DosageStore.open(root) as store:
        codes = store.read_codes(0, store.n_variants, training)
    assert np.array_equal(codes, _all_codes(milli_by_half)[:, training])
    signed = codes.astype(np.int64) - 127
    count = training.size
    sums, squares = signed.sum(axis=1), (signed * signed).sum(axis=1)
    scaled_variance = count * squares - sums * sums
    standardized = (count * signed - sums[:, None]) / np.sqrt(scaled_variance.astype(np.float64))[:, None]
    dosage = codes / 127.0
    reference = (dosage - dosage.mean(axis=1, keepdims=True)) / dosage.std(axis=1, keepdims=True)
    # The integer route rounds only in its square root and division. The reference's mean and
    # standard deviation are float sums of count terms of size at most 2, each off by at most
    # count * eps * 2 (Higham's gamma_n), which moves z by that much times (1 + |z|) / sd.
    sums_error = (2 * count + 4) * np.finfo(np.float64).eps * 2.0
    spread = dosage.std(axis=1, keepdims=True)
    bound = sums_error * (1.0 + np.abs(reference).max(axis=1, keepdims=True)) / spread
    assert np.all(np.abs(standardized - reference) <= bound)


def test_transcoding_rejects_a_sidecar_that_disagrees_with_the_codes(
    two_half_store: tuple[Path, list[dict[str, np.ndarray]]],
) -> None:
    root, _ = two_half_store
    sums, _ = open_column(statistic_column_directory(root, 1, "chr22", "sum_code"), writable=True)
    sums[7] += 1
    sums.flush()
    with pytest.raises(ValueError, match="sum_code"):
        transcode_store(root, root.parent / "cache", codec="raw", budget=_budget())


def test_stage0_streams_a_store_exactly_like_the_in_memory_source(
    two_half_store: tuple[Path, list[dict[str, np.ndarray]]],
) -> None:
    root, milli_by_half = two_half_store
    codes = _all_codes(milli_by_half)
    with DosageStore.open(root) as store:
        source = StoreTileSource(store)
        assert source.chromosomes() == ["chr21", "chr22"] and source.sample_count == 60
        assert np.array_equal(source.store_rows("chr22"), np.arange(300, 450))
        reference = InMemoryTileSource(
            codes={"chr21": codes[:300], "chr22": codes[300:]},
            groups={name: source.unsplittable_groups(name) for name in source.chromosomes()},
        )
        results = []
        for tile_source in (source, reference):
            layout = build_sample_layout(np.arange(60) % 2)
            plan = plan_genotype_pass(64)
            buffer = HostGenotypeBuffer(layout, plan.capacity_rows, None, worker_count=2)
            blocks = []
            summary = run_genotype_pass(tile_source, layout, [buffer], plan, lambda block, _: blocks.append(block))
            buffer.close()
            results.append((summary, blocks))
    (store_summary, store_blocks), (memory_summary, memory_blocks) = results
    for chromosome in ("chr21", "chr22"):
        assert np.array_equal(store_summary.chromosomes[chromosome].boundaries, memory_summary.chromosomes[chromosome].boundaries)
    assert len(store_blocks) == len(memory_blocks)
    for from_store, from_memory in zip(store_blocks, memory_blocks):
        assert (from_store.chromosome, from_store.start, from_store.stop) == (from_memory.chromosome, from_memory.start, from_memory.stop)
        assert np.array_equal(from_store.grams, from_memory.grams) and np.array_equal(from_store.sums, from_memory.sums)


def test_stage0_accumulation_bounds_are_exact_and_tight() -> None:
    """|s| <= 127 for every stored code, so int32 sums of s_i s_j are exact over INT32_EXACT_ROWS
    samples and fp32 sums over FLOAT32_EXACT_ROWS, and one more sample breaks each."""
    assert encode_dosage_milli(np.array([0, 2000])).tolist() == [127 - SIGNED_CODE_OFFSET, 127 + SIGNED_CODE_OFFSET]
    worst_product = SIGNED_CODE_OFFSET * SIGNED_CODE_OFFSET
    assert (INT32_EXACT_ROWS, FLOAT32_EXACT_ROWS) == (133_144, 1_040)
    integer_products = np.full(INT32_EXACT_ROWS + 1, worst_product, dtype=np.int32)
    integer_running = np.add.accumulate(integer_products, dtype=np.int32)
    assert int(integer_running[INT32_EXACT_ROWS - 1]) == worst_product * INT32_EXACT_ROWS
    assert int(integer_running[INT32_EXACT_ROWS]) != worst_product * (INT32_EXACT_ROWS + 1)
    float_products = np.full(FLOAT32_EXACT_ROWS + 1, worst_product, dtype=np.float32)
    float_running = np.add.accumulate(float_products, dtype=np.float32)
    assert float(float_running[FLOAT32_EXACT_ROWS - 1]) == worst_product * FLOAT32_EXACT_ROWS
    assert float(float_running[FLOAT32_EXACT_ROWS]) != worst_product * (FLOAT32_EXACT_ROWS + 1)

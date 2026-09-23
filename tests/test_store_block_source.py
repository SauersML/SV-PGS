"""Stage 2 tiles streamed from a dosage store equal tiles built from the gathered codes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sv_pgs.code_products import CodeBlockTile
from sv_pgs.compute_budget import ComputeBudget, _try_import_cupy
from sv_pgs.dosage_store import DosageStore, encode_dosage_milli
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from tests.test_dosage_store import _two_half_dosage, _write_store

cupy = _try_import_cupy()
FLOAT64_ROUNDING = float(np.finfo(np.float64).eps) / 2
WORKSPACE_BYTES = 1 << 30
# ascending, gapped rows (inactive and tied rows skipped), one block ending a chromosome
BLOCK_ROWS = [np.array([3, 5, 6, 10, 40]), np.array([41, 44, 47, 99, 298, 299]), np.array([300, 301, 449])]


def _budget(device_kind: str) -> ComputeBudget:
    on_device = device_kind == "cuda"
    return ComputeBudget(
        device_kind=device_kind,
        device_ids=(0,) if on_device else (),
        device_names=("test",) if on_device else (),
        device_bytes=(1 << 33,) if on_device else (),
        device_compute_capabilities=((8, 6),) if on_device else (),
        host_bytes=1 << 30,
        cpu_threads=2,
    )


def _signed_store_codes(milli_by_half: list[dict[str, np.ndarray]]) -> np.ndarray:
    codes = np.hstack([np.vstack([encode_dosage_milli(milli) for milli in half.values()]) for half in milli_by_half])
    return (codes.astype(np.int16) - SIGNED_CODE_OFFSET).astype(np.int8)


def _source(root: Path, device_kind: str) -> tuple[StoreGenotypeBlockSource, np.ndarray, np.ndarray, np.ndarray]:
    store = DosageStore.open(root)
    signed = _signed_store_codes(_two_half_dosage())
    rows = np.concatenate(BLOCK_ROWS)
    values = signed[rows].astype(np.float64)
    means, scales = values.mean(axis=1), values.std(axis=1)
    indices = np.split(np.arange(rows.shape[0]), np.cumsum([block.shape[0] for block in BLOCK_ROWS])[:-1])
    source = StoreGenotypeBlockSource(store, BLOCK_ROWS, indices, means, scales, _budget(device_kind), WORKSPACE_BYTES)
    return source, signed, means, scales


def _bits(array) -> np.ndarray:
    host = cupy.asnumpy(array) if cupy is not None and not isinstance(array, np.ndarray) else array
    return np.ascontiguousarray(host).view(np.uint64)


@pytest.mark.parametrize("codec", ["raw", "zstd", "rowdict"])
def test_cpu_streamed_tiles_equal_tiles_of_the_gathered_codes(tmp_path: Path, codec: str) -> None:
    _write_store(tmp_path / "store", _two_half_dosage(), codec)
    source, signed, means, scales = _source(tmp_path / "store", "cpu")
    rng = np.random.default_rng(1)
    left = rng.standard_normal((source.sample_count, 3))
    offsets = np.cumsum([0] + [block.shape[0] for block in BLOCK_ROWS])
    seen = []
    for block_index, tile in source.iter_tiles():
        columns = slice(int(offsets[block_index]), int(offsets[block_index + 1]))
        expected = CodeBlockTile(signed[BLOCK_ROWS[block_index]], means[columns], scales[columns], np, WORKSPACE_BYTES)
        right = rng.standard_normal((tile.variant_count, 2))
        assert np.array_equal(_bits(tile.rmatmat(left)), _bits(expected.rmatmat(left)))
        assert np.array_equal(_bits(tile.matmat(right)), _bits(expected.matmat(right)))
        seen.append(block_index)
    assert seen == [0, 1, 2]
    assert source.sample_count == signed.shape[1]


def test_blocks_must_hold_ascending_distinct_rows(tmp_path: Path) -> None:
    _write_store(tmp_path / "store", _two_half_dosage(), "raw")
    store = DosageStore.open(tmp_path / "store")
    with pytest.raises(ValueError, match="ascending"):
        StoreGenotypeBlockSource(store, [np.array([4, 4])], [np.arange(2)], np.zeros(2), np.ones(2), _budget("cpu"), WORKSPACE_BYTES)


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
@pytest.mark.parametrize("codec", ["raw", "zstd", "rowdict"])
def test_cuda_streamed_tiles_equal_tiles_of_the_gathered_codes(tmp_path: Path, codec: str) -> None:
    _write_store(tmp_path / "store", _two_half_dosage(), codec)
    source, signed, means, scales = _source(tmp_path / "store", "cuda")
    rng = np.random.default_rng(2)
    left = cupy.asarray(rng.standard_normal((source.sample_count, 3)))
    offsets = np.cumsum([0] + [block.shape[0] for block in BLOCK_ROWS])
    image = cupy.zeros((source.sample_count, 2))
    expected_image = cupy.zeros((source.sample_count, 2))
    magnitude = np.zeros((source.sample_count, 2))
    operand = None
    for block_index, tile in source.iter_tiles():
        columns = slice(int(offsets[block_index]), int(offsets[block_index + 1]))
        expected = CodeBlockTile(
            cupy.asarray(signed[BLOCK_ROWS[block_index]]), cupy.asarray(means[columns]), cupy.asarray(scales[columns]), cupy, WORKSPACE_BYTES
        )
        operand = tile.sample_operand(left, FLOAT64_ROUNDING) if operand is None else operand
        right = cupy.asarray(rng.standard_normal((tile.variant_count, 2)))
        assert np.array_equal(_bits(tile.rmatmat(operand)), _bits(expected.rmatmat(left)))
        tile.accumulate_matmat(right, image, FLOAT64_ROUNDING)
        expected_image += expected.matmat(right)
        # The block's terms' sizes: |s_ij| |r_jk / scale_j| and the centring's |mean_j| |r_jk / scale_j|.
        weights = np.abs(cupy.asnumpy(right)) / scales[columns, None]
        magnitude += np.abs(signed[BLOCK_ROWS[block_index]].astype(np.float64)).T @ weights + (np.abs(means[columns]) @ weights)[None, :]
    # Two float64 sums of the same terms in different orders (the digit kernel's and the device GEMM's: an H100's FP64
    # tensor cores order a dot product differently from an A40's): each is within gamma_m of the exact sum, m the
    # longest chain of roundings (every block's rows, its centring and scaling, and the image's sum over blocks), so
    # they are within twice that of each other (Higham 2002, section 3.1). Summation order is not a contract.
    chain = max(block.shape[0] for block in BLOCK_ROWS) + len(("centring", "scaling")) + len(BLOCK_ROWS)
    gamma = chain * FLOAT64_ROUNDING / (1.0 - chain * FLOAT64_ROUNDING)
    assert np.all(np.abs(cupy.asnumpy(image) - cupy.asnumpy(expected_image)) <= 2.0 * gamma * magnitude)


def test_a_stage_2_read_over_streamed_tiles_equals_the_per_block_products(tmp_path: Path) -> None:
    from sv_pgs.exact_polish import _Device, _Reads

    _write_store(tmp_path / "store", _two_half_dosage(), "zstd")
    source, signed, means, scales = _source(tmp_path / "store", "cpu")
    rng = np.random.default_rng(3)
    left = rng.standard_normal((source.sample_count, 4))
    rights = [rng.standard_normal((block.shape[0], 4)) for block in BLOCK_ROWS]
    seen = []

    def local(block_index, variants, tile, products):
        seen.append(products)
        return rights[block_index]

    image = _Reads(source, _Device(np)).sweep(left, local, 4)
    offsets = np.cumsum([0] + [block.shape[0] for block in BLOCK_ROWS])
    expected = np.zeros_like(image)
    for block_index, rows in enumerate(BLOCK_ROWS):
        columns = slice(int(offsets[block_index]), int(offsets[block_index + 1]))
        tile = CodeBlockTile(signed[rows], means[columns], scales[columns], np, WORKSPACE_BYTES)
        assert np.array_equal(_bits(seen[block_index]), _bits(tile.rmatmat(left)))
        expected += tile.matmat(rights[block_index])
    assert np.array_equal(_bits(image), _bits(expected))


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
@pytest.mark.parametrize("codec", ["zstd", "rowdict"])
def test_cuda_reads_after_the_first_come_from_the_resident_codes(tmp_path: Path, codec: str, monkeypatch) -> None:
    # The first read keeps every block's signed codes on the device; a second read builds the same tiles from them
    # without touching the store.
    _write_store(tmp_path / "store", _two_half_dosage(), codec)
    source, _signed, _means, _scales = _source(tmp_path / "store", "cuda")
    rng = np.random.default_rng(4)
    rights = [cupy.asarray(rng.standard_normal((block.shape[0], 2))) for block in BLOCK_ROWS]

    def image() -> np.ndarray:
        total = cupy.zeros((source.sample_count, 2))
        for block_index, tile in source.iter_tiles():
            tile.accumulate_matmat(rights[block_index], total, FLOAT64_ROUNDING)
        return cupy.asnumpy(total)

    first = image()
    assert source._resident_complete and source._resident is not None

    def no_read(*_args, **_kwargs):
        raise AssertionError("a resident read touched the store")

    monkeypatch.setattr(source._store, "iter_codes", no_read)
    monkeypatch.setattr(source._store, "read_codes_to_device", no_read)
    assert np.array_equal(_bits(image()), _bits(first))


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
@pytest.mark.parametrize("codec", ["zstd", "rowdict"])
def test_cuda_resident_codes_evicted_mid_read_stream_the_blocks_left(tmp_path: Path, codec: str) -> None:
    # The resident codes are a cache of the shared ledger: evicted during a read, the blocks not yet reached are read
    # from the store, the image is the same, and an evicted set is never admitted again.
    _write_store(tmp_path / "store", _two_half_dosage(), codec)
    source, _signed, _means, _scales = _source(tmp_path / "store", "cuda")
    rng = np.random.default_rng(5)
    rights = [cupy.asarray(rng.standard_normal((block.shape[0], 2))) for block in BLOCK_ROWS]

    def image(evict_after: int | None = None) -> np.ndarray:
        total = cupy.zeros((source.sample_count, 2))
        for block_index, tile in source.iter_tiles():
            tile.accumulate_matmat(rights[block_index], total, FLOAT64_ROUNDING)
            if block_index == evict_after:
                (lease,) = [lease for lease in source._broker.leases if lease.evict == source._drop_resident]
                source._broker._evict(lease)
        return cupy.asnumpy(total)

    first = image()
    assert source._resident_complete and source._resident is not None
    assert np.array_equal(_bits(image(evict_after=0)), _bits(first))
    assert source._resident is None and source._resident_dropped
    assert np.array_equal(_bits(image()), _bits(first)) and source._resident is None


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
def test_inside_a_cuda_scope_every_allocation_is_charged_before_it_is_made() -> None:
    from sv_pgs.memory_broker import device_pool, memory_scope

    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    capacity = int(pool.used_bytes()) + (64 << 20)
    budget = ComputeBudget(
        device_kind="cuda", device_ids=(0,), device_names=("test",), device_bytes=(capacity,), device_compute_capabilities=((8, 6),),
        host_bytes=1 << 30, cpu_threads=2,
    )
    dropped = []
    cache = {}
    with memory_scope(budget) as broker:
        held = cupy.zeros(4 << 20, dtype=cupy.uint8)
        cache["array"] = cupy.zeros(32 << 20, dtype=cupy.uint8)
        lease = broker.admit(device_pool(0), 32 << 20, "a test cache", lambda: (dropped.append(1), cache.clear()), allocated=True)
        assert lease is not None
        # 40 MB more fits only once the cache is dropped: the allocator evicts it before allocating.
        more = cupy.zeros(40 << 20, dtype=cupy.uint8)
        assert dropped == [1] and broker.held(device_pool(0)) <= capacity
        with pytest.raises(MemoryError):
            cupy.zeros(64 << 20, dtype=cupy.uint8)
        del held, more
    # Outside the scope the default allocator is back.
    cupy.zeros(1 << 20, dtype=cupy.uint8)

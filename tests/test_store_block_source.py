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


@pytest.mark.parametrize("codec", ["raw", "zstd"])
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
@pytest.mark.parametrize("codec", ["raw", "zstd"])
def test_cuda_streamed_tiles_equal_tiles_of_the_gathered_codes(tmp_path: Path, codec: str) -> None:
    _write_store(tmp_path / "store", _two_half_dosage(), codec)
    source, signed, means, scales = _source(tmp_path / "store", "cuda")
    rng = np.random.default_rng(2)
    left = cupy.asarray(rng.standard_normal((source.sample_count, 3)))
    offsets = np.cumsum([0] + [block.shape[0] for block in BLOCK_ROWS])
    image = cupy.zeros((source.sample_count, 2))
    expected_image = cupy.zeros((source.sample_count, 2))
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
    assert np.array_equal(_bits(image), _bits(expected_image))

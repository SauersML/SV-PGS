from pathlib import Path

import numpy as np
import pytest

from sv_pgs import rowdict_codec
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

"""Stage 0's sums for the fit (``genotype_statistics._project_block``): each block's lag profile and its squared Grams'
sums, taken while Stage 0 holds the block, are what their readers would take from the stored Grams, to the bit; the
EB start from them is the start that read every Gram again; and the summary band laid out from the lag profiles and
written when needed is, written whole, the band copied out of every block, byte for byte."""

from __future__ import annotations

import filecmp
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.stage0_bench import copied_band_store, moment_starts_read_all
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig
from sv_pgs.full_data_fit import block_grams, moment_starts
from sv_pgs.genotype_statistics import compute_genotype_statistics
from sv_pgs.gram_space import _BAND_FILES, _BAND_INDEX, BandStore, summary_band
from sv_pgs.lag_profiles import LAG_PROFILE_ROWS
from sv_pgs.marginal_variances import ld_lag_profiles
from sv_pgs.scale_mixture_ep import device_scope
from tests.stage0_support import InMemoryTileSource, mosaic_codes

_SAMPLES = 160
_BLOCK_CAP = 1024
_LEVEL = 1.0 / 64


def _statistics(directory: Path):
    """Two chromosomes of mosaic LD, the first wider than a projection panel's rows in some block, on the threaded host
    buffer, with a second target so the scores are a matrix."""
    rng = np.random.default_rng(31)
    counts = {"chr1": 1500, "chr2": 700}
    codes = {name: mosaic_codes(rng, _SAMPLES, count) for name, count in counts.items()}
    source = InMemoryTileSource(codes=codes, groups={name: np.arange(count, dtype=np.int64) for name, count in counts.items()})
    covariates = np.column_stack([np.ones(_SAMPLES), rng.normal(size=(_SAMPLES, 2))])
    budget = ComputeBudget(device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(),
                           host_bytes=1 << 31, cpu_threads=3)
    statistics = compute_genotype_statistics(source, np.arange(_SAMPLES), covariates, rng.normal(size=(_SAMPLES, 2)), ModelConfig(),
                                             budget, _BLOCK_CAP, directory)
    return statistics, rng


def test_stage0_keeps_the_sums_its_readers_take_from_the_stored_grams(tmp_path: Path) -> None:
    statistics, rng = _statistics(tmp_path / "ld")
    ld = statistics.ld
    widths = np.diff(ld.block_boundaries)
    assert widths.max() > LAG_PROFILE_ROWS and ld.block_count >= 3
    # The lag profiles, as the EP windows' extent sums them from the stored Grams.
    excess, pairs = ld_lag_profiles(block_grams(statistics), statistics.sample_count, 1 << 30)
    stored_excess, stored_pairs = ld.lag_profiles()
    assert np.array_equal(stored_excess, excess) and np.array_equal(stored_pairs, pairs)
    # The squared Grams' sums, as the EB start squares them.
    for block in range(ld.block_count):
        square = ld.read_gram(block).astype(np.float64)
        square *= square
        assert np.array_equal(ld.square_columns(block), square.sum(axis=0)) and ld.square_sum(block) == float(square.sum())
        if ld.has_adjacent(block):
            coupling = ld.read_adjacent(block).astype(np.float64)
            coupling *= coupling
            assert np.array_equal(ld.adjacent_square_rows(block), coupling.sum(axis=1))
            assert np.array_equal(ld.adjacent_square_columns(block), coupling.sum(axis=0))
            assert ld.adjacent_square_sum(block) == float(coupling.sum())
    # The EB start from those sums is the read-everything start as a device ran it, to the bit. Its one sum not Stage
    # 0's, sum_j u_j G_jj, is a dot product with the Gram's diagonal, which a device's copy brought back contiguous, as
    # the stored diagonal is; on the host it was a strided view, which BLAS sums in another order (a few units in the
    # last place apart), so the read-everything start runs here on a host module whose diagonal comes back as a
    # device's does.
    prior = SimpleNamespace(log_variance_offset=rng.normal(scale=0.5, size=np.asarray(statistics.active_rows).shape[0]))
    device_like = SimpleNamespace(asarray=np.asarray, float64=np.float64, diag=lambda values: np.ascontiguousarray(np.diag(values)))
    with device_scope(device_like):
        read_all = moment_starts_read_all(statistics, prior)
    assert moment_starts(statistics, prior) == read_all


@pytest.mark.parametrize("null_scale", [1.0, 1.3])
def test_the_summary_band_is_written_when_needed_and_whole_is_the_copied_band(tmp_path: Path, null_scale: float) -> None:
    statistics, _rng = _statistics(tmp_path / "ld")
    ld = statistics.ld
    band = summary_band(ld, statistics.sample_count, tmp_path / "band", _LEVEL, null_scale)
    copied = copied_band_store(ld, statistics.sample_count, tmp_path / "copied", 1 << 30, _LEVEL, np, null_scale)
    assert band.extents == copied.extents and np.array_equal(band.widths, copied.widths)
    assert any(entry["parts"] > 1 for entry in band.extents) and band.block_count > 2
    # Only what pass_costs reads is written: part 0's Gram and part 1's Gram with it.
    assert 0 < band.written_bytes < band.stored_bytes
    last = band.block_count - 1
    unopened = BandStore(tmp_path / "band")
    with pytest.raises(RuntimeError, match="has not written"):
        unopened.read_gram(last)
    np.testing.assert_array_equal(band.read_gram(0), copied.read_gram(0))
    np.testing.assert_array_equal(band.read_adjacent(1), copied.read_adjacent(1))
    # A read of a part not yet written writes its Stage 0 block's parts first.
    np.testing.assert_array_equal(band.read_gram(last), copied.read_gram(last))
    band.complete()
    assert band.written_bytes == band.stored_bytes
    for file_name in _BAND_FILES.values():
        assert filecmp.cmp(tmp_path / "band" / file_name, tmp_path / "copied" / file_name, shallow=False)
    index = json.loads((tmp_path / "band" / _BAND_INDEX).read_text())
    copied_index = json.loads((tmp_path / "copied" / _BAND_INDEX).read_text())
    assert all(index[key] == copied_index[key] for key in copied_index) and index["unwritten"] == []
    # Opened again from its directory alone, the written band reads as it was written.
    reopened = BandStore(tmp_path / "band")
    for part in range(reopened.block_count):
        np.testing.assert_array_equal(reopened.read_gram(part), copied.read_gram(part))
        if copied.has_adjacent(part):
            np.testing.assert_array_equal(reopened.read_adjacent(part), copied.read_adjacent(part))


def test_the_stage0_benchmark_passes_its_checks_on_the_host(tmp_path: Path) -> None:
    from benchmarks.stage0_bench import main

    report = main(["--tiny", "--device", "cpu", "--out", str(tmp_path)])
    assert report["checks"] and all(report["checks"].values()), report["checks"]


def test_the_cuda_buffer_hands_a_host_array_back_from_to_host() -> None:
    # Stage 0's lag profile is summed on the host (lag_profiles.lag_profile), and _project_block sends every array it
    # returns through the buffer's to_host. CuPy's ascontiguousarray refuses a NumPy array, which failed both arms of
    # round m1 at Stage 0 ("Argument 'a' has incorrect type"); the fake cupy here refuses one the same way.
    from sv_pgs.genotype_buffers import CudaGenotypeBuffer

    def refuse(array):
        raise TypeError("Argument 'a' has incorrect type (expected cupy._core.core._ndarray_base, got numpy.ndarray)")

    buffer = object.__new__(CudaGenotypeBuffer)
    buffer._cupy = SimpleNamespace(ascontiguousarray=refuse)
    host = np.arange(12.0).reshape(3, 4)[:, ::2]
    returned = buffer.to_host(host)
    assert isinstance(returned, np.ndarray) and returned.flags.c_contiguous
    np.testing.assert_array_equal(returned, host)

"""Stage 0's kept training codes (``sv_pgs.training_codes``): kept from Stage 0's own tiles, they are the store's codes
bit for bit, Stage 2's reads (the block source's tiles, the devices' resident codes, the far field's columns) take them
without reading the store, a read whose codes are evicted takes the blocks left from the store, and a fit is the fit
without them, bit for bit."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from sv_pgs import training_codes
from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
from sv_pgs.memory_broker import HOST, memory_scope
from sv_pgs.restart_store import CodeCache, RestartStore
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from sv_pgs.training_codes import CodeTee, TrainingCodes
from tests.test_dosage_store import _two_half_dosage, _write_store
from tests.test_eb_monitor import _checkpoints, _same, _saved
from tests.test_full_data_fit import _BLOCK_CAP, _TRAINING, _store
from tests.test_full_data_fit import _budget as _fit_budget
from tests.test_stage2_wiring import _budget as _wiring_budget
from tests.test_store_block_source import BLOCK_ROWS, FLOAT64_ROUNDING, WORKSPACE_BYTES, _bits, _budget

cupy = _try_import_cupy()
# Stage 0's tiles in these tests: a few consecutive candidates each, so a block's rows span several tiles.
TILE_ROWS = 64


def _signed(codes: np.ndarray) -> np.ndarray:
    """The host's signed codes of store codes: the int8 view less SIGNED_CODE_OFFSET, narrowed as ``_iter_store`` does."""
    out = np.empty(codes.shape, dtype=np.int8)
    np.subtract(np.ascontiguousarray(codes).view(np.int8), np.int8(SIGNED_CODE_OFFSET), out=out)
    return out


def _kept(store: DosageStore, columns: np.ndarray, chromosomes: int | None = None) -> TrainingCodes:
    """Every store record's codes on ``columns``, kept as Stage 0's pass keeps them: tile by tile, each tile every sample
    of a few consecutive candidates of one chromosome (``chromosomes``: only the first so many)."""
    codes = TrainingCodes(np.arange(store.n_variants, dtype=np.int64), np.asarray(columns, dtype=np.int64))
    bounds = np.asarray(store.chromosome_starts, dtype=np.int64)
    for chromosome in range(len(bounds) - 1 if chromosomes is None else chromosomes):
        for start in range(int(bounds[chromosome]), int(bounds[chromosome + 1]), TILE_ROWS):
            stop = min(start + TILE_ROWS, int(bounds[chromosome + 1]))
            codes.capture(np.arange(start, stop, dtype=np.int64), store.read_codes(start, stop))
    return codes


def _source(
    store: DosageStore, device_kind: str, sample_indices: np.ndarray | None, codes: TrainingCodes | None, code_cache: CodeCache | None = None,
) -> StoreGenotypeBlockSource:
    """``test_store_block_source``'s blocks (gapped rows, one ending a chromosome) on ``sample_indices``, reading from
    ``codes`` where given, and keeping each block's codes as a restart unit of ``code_cache`` where given."""
    rows = np.concatenate(BLOCK_ROWS)
    values = _signed(store.read_codes(0, store.n_variants))[rows].astype(np.float64)
    if sample_indices is not None:
        values = values[:, sample_indices]
    indices = np.split(np.arange(rows.shape[0]), np.cumsum([block.shape[0] for block in BLOCK_ROWS])[:-1])
    return StoreGenotypeBlockSource(
        store, BLOCK_ROWS, indices, values.mean(axis=1), values.std(axis=1), _budget(device_kind), WORKSPACE_BYTES, sample_indices,
        code_cache=code_cache, codes=codes,
    )


def _products(source: StoreGenotypeBlockSource, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each block's rmatmat and matmat of one read, as bits."""
    xp = source.array_module
    generator = np.random.default_rng(seed)
    left = xp.asarray(generator.standard_normal((source.sample_count, 3)))
    found = []
    for _block_index, tile in source.iter_tiles():
        right = xp.asarray(generator.standard_normal((tile.variant_count, 2)))
        found.append((_bits(tile.rmatmat(left)), _bits(tile.matmat(right))))
    return found


def _identical(found: list[tuple[np.ndarray, np.ndarray]], expected: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    return len(found) == len(expected) and all(
        np.array_equal(left, right) for pair, other in zip(found, expected) for left, right in zip(pair, other)
    )


def _no_read(*_arguments, **_keywords):
    raise AssertionError("a read from the kept codes touched the store")


def test_a_kept_code_is_the_signed_code_the_reads_make_and_undoes_to_the_store_s() -> None:
    """Every one of the 256 codes is kept as the host's subtraction (and the device's gather) narrows it, two's
    complement, and ``unsigned`` gives the store's own code back; a column subset keeps its columns in order."""
    tile = np.arange(256, dtype=np.uint8).reshape(4, 64)
    codes = TrainingCodes(np.array([2, 3, 5, 9], dtype=np.int64), np.arange(64, dtype=np.int64))
    codes.capture(np.array([2, 3, 5, 9], dtype=np.int64), tile)
    assert codes.off is None and np.array_equal(codes.codes, _signed(tile))
    np.testing.assert_array_equal(codes.unsigned(np.array([2, 3, 5, 9])), tile)
    np.testing.assert_array_equal(codes.unsigned(np.array([5])), tile[2:3])
    columns = np.array([0, 7, 8, 63], dtype=np.int64)
    subset = TrainingCodes(np.array([2, 3, 5, 9], dtype=np.int64), columns)
    subset.capture(np.array([2, 3], dtype=np.int64), tile[:2])
    subset.capture(np.array([5, 9], dtype=np.int64), tile[2:])
    np.testing.assert_array_equal(subset.unsigned(np.array([2, 3, 5, 9])), tile[:, columns])
    # A row that is no candidate, or one no tile filled, is not served.
    assert subset.positions(np.array([4])) is None and subset.unsigned(np.array([10])) is None
    unfilled = TrainingCodes(np.array([2, 3, 5, 9], dtype=np.int64), columns)
    unfilled.capture(np.array([2, 3], dtype=np.int64), tile[:2])
    assert unfilled.positions(np.array([2, 3])) is not None and unfilled.positions(np.array([3, 5])) is None


def test_a_tile_the_codes_cannot_take_turns_them_off_and_never_fails_stage0(capsys: pytest.CaptureFixture[str]) -> None:
    codes = TrainingCodes(np.arange(10, dtype=np.int64), np.arange(4, dtype=np.int64))
    codes.capture(np.array([3, 7], dtype=np.int64), np.zeros((2, 8), dtype=np.uint8))
    assert codes.off is not None and codes.positions(np.array([3])) is None
    assert "training codes: off: a Stage 0 tile could not be kept" in capsys.readouterr().err


def test_stage0_through_the_tee_is_stage0_and_keeps_every_candidate_s_training_codes(tmp_path: Path) -> None:
    """Stage 0 read through ``CodeTee`` is Stage 0 read from the store (its statistics and LD Grams bit for bit), and
    the codes then hold every candidate's codes on the training samples, gapped candidates included."""
    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    training = np.arange(0, _TRAINING, dtype=np.int64)
    covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    # Gapped candidates: every seventh record left out.
    candidates = np.flatnonzero(np.arange(store.n_variants) % 7 != 3).astype(np.int64)
    codes = TrainingCodes(candidates, training)
    tee = compute_genotype_statistics(
        CodeTee(store, candidates, [codes]), training, covariates, targets[training, None], ModelConfig(), _fit_budget(), _BLOCK_CAP, tmp_path / "tee",
    )
    plain = compute_genotype_statistics(
        DosageStoreTileSource(store, candidates), training, covariates, targets[training, None], ModelConfig(), _fit_budget(), _BLOCK_CAP, tmp_path / "plain",
    )
    for name in ("active_rows", "means", "scales", "allele_frequency", "block_of_reduced", "covariate_target", "target_gram"):
        np.testing.assert_array_equal(getattr(tee, name), getattr(plain, name), err_msg=name)
    np.testing.assert_array_equal(tee.tie_map.kept_indices, plain.tie_map.kept_indices)
    assert tee.ld.block_count == plain.ld.block_count
    for block in range(plain.ld.block_count):
        np.testing.assert_array_equal(tee.ld.read_gram(block), plain.ld.read_gram(block))
    assert codes.off is None and bool(np.all(codes.filled))
    np.testing.assert_array_equal(codes.positions(candidates), np.arange(candidates.shape[0]))
    np.testing.assert_array_equal(codes.unsigned(candidates), store.read_codes(0, store.n_variants)[candidates][:, training])


@pytest.mark.parametrize("codec", ["raw", "zstd", "rowdict"])
@pytest.mark.parametrize("selected", [False, True])
def test_host_tiles_from_kept_codes_are_the_store_s_tiles_and_read_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codec: str, selected: bool) -> None:
    """On the host every read takes its blocks from the kept codes, never the store, and each block's products are a
    store read's bit for bit, on every sample or on a training subset."""
    _write_store(tmp_path / "store", _two_half_dosage(), codec)
    store = DosageStore.open(tmp_path / "store")
    sample_indices = np.arange(0, store.n_samples, 2, dtype=np.int64) if selected else None
    streamed = _products(_source(store, "cpu", sample_indices, None), 3)
    columns = np.arange(store.n_samples, dtype=np.int64) if sample_indices is None else sample_indices
    source = _source(store, "cpu", sample_indices, _kept(store, columns))
    monkeypatch.setattr(store, "iter_codes", _no_read)
    monkeypatch.setattr(store, "read_codes", _no_read)
    for _read in range(2):
        assert _identical(_products(source, 3), streamed)
    store.close()


def test_codes_evicted_mid_read_leave_the_blocks_left_to_the_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The codes are a cache of the host ledger: evicted between two blocks of a read, the blocks left are read from the
    store, every product unchanged, and later reads are the store's."""
    _write_store(tmp_path / "store", _two_half_dosage(), "zstd")
    store = DosageStore.open(tmp_path / "store")
    streamed = _products(_source(store, "cpu", None, None), 4)
    with memory_scope(_budget("cpu")) as broker:
        codes = _kept(store, np.arange(store.n_samples, dtype=np.int64))
        source = _source(store, "cpu", None, codes)
        generator = np.random.default_rng(4)
        left = generator.standard_normal((source.sample_count, 3))
        found = []
        for block_index, tile in source.iter_tiles():
            right = generator.standard_normal((tile.variant_count, 2))
            found.append((_bits(tile.rmatmat(left)), _bits(tile.matmat(right))))
            if block_index == 0:
                broker.evict_idle(HOST)
                assert codes.off is not None
        assert _identical(found, streamed)
        assert _identical(_products(source, 4), streamed)
    assert capsys.readouterr().err.count("training codes: off: the ledger evicted them") == 1
    store.close()


def test_codes_that_are_not_the_source_s_are_never_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Codes of other samples serve no source; codes that hold only some of a source's records (a Stage 0 that read
    only part of the store) serve none of its reads, which are the store's."""
    _write_store(tmp_path / "store", _two_half_dosage(), "raw")
    store = DosageStore.open(tmp_path / "store")
    selected = np.arange(0, store.n_samples, 2, dtype=np.int64)
    streamed = _products(_source(store, "cpu", selected, None), 5)
    other = _source(store, "cpu", selected, _kept(store, selected + 1))
    assert other._codes is None and _identical(_products(other, 5), streamed)
    partial = _source(store, "cpu", selected, _kept(store, selected, chromosomes=1))
    assert _identical(_products(partial, 5), streamed) and partial._codes is None
    assert "training codes: Stage 0's pass did not keep every block's records; Stage 2 reads the store" in capsys.readouterr().err
    store.close()


def test_the_far_field_from_kept_codes_is_the_far_field_from_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``far_field_scale`` given the kept codes reads no record from the store and measures the same scale, error and
    pairs, bit for bit."""
    from sv_pgs.gram_space import far_field_scale

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING, dtype=np.int64)
    covariates = np.column_stack([np.ones(_TRAINING), covariate[training]])
    candidates = np.arange(store.n_variants, dtype=np.int64)
    codes = TrainingCodes(candidates, training)
    statistics = compute_genotype_statistics(
        CodeTee(store, candidates, [codes]), training, covariates, targets[training, None], ModelConfig(), _fit_budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    ld = statistics.ld
    starts = np.asarray(ld.block_boundaries, dtype=np.int64)
    linked = np.array([ld.has_adjacent(block + 1) for block in range(ld.block_count - 1)], dtype=bool)
    plain = far_field_scale(store, training, covariates, statistics, starts, linked, 16, 3)
    monkeypatch.setattr(store, "read_codes", _no_read)
    assert far_field_scale(store, training, covariates, statistics, starts, linked, 16, 3, codes=codes) == plain
    assert plain[2] > 0


@pytest.mark.parametrize("route", ["samples", "gram"])
def test_a_fit_from_the_kept_codes_is_the_fit_from_the_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    """A fit with the codes kept (CODE_CACHE_SWITCH unset) is bit for bit the fit without them (at 0): every oracle
    call's ELBO, noise and hyperparameters, its checkpoint (key, path and content) and its saved model. And with them
    kept, nothing after Stage 0 reads the store in the group's fit (the block source's reads, the far field's columns),
    where without them both read it."""
    from sv_pgs import dosage_store, fit_model, full_data_fit, stage2_wiring
    from sv_pgs.artifact import save_model

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    monkeypatch.setattr(stage2_wiring, "pass_costs", (lambda band, source, xp: (0.0, np.inf)) if route == "samples" else (lambda band, source, xp: (np.inf, 0.0)))
    monkeypatch.setattr(fit_model, "DRAW_COUNT", 16)
    samples = store.n_samples
    held_out = np.arange(samples) >= samples * 4 // 5
    # Each store read, with the phase of the group's fit it was made in.
    phase = ["stage0"]
    reads: list[tuple[str, str]] = []
    for method in ("read_codes", "iter_codes", "read_codes_to_device"):
        original = getattr(dosage_store.DosageStore, method)

        def counted(self, *arguments, _original=original, _method=method, **keywords):
            reads.append((phase[0], _method))
            return _original(self, *arguments, **keywords)

        monkeypatch.setattr(dosage_store.DosageStore, method, counted)
    statistics_of = stage2_wiring.compute_genotype_statistics
    group_fit = stage2_wiring._fit_group

    def stage0(*arguments, **keywords):
        found = statistics_of(*arguments, **keywords)
        phase[0] = "stage2"
        return found

    def fit_group(*arguments, **keywords):
        phase[0] = "stage0"
        try:
            return group_fit(*arguments, **keywords)
        finally:
            phase[0] = "after"

    monkeypatch.setattr(stage2_wiring, "compute_genotype_statistics", stage0)
    monkeypatch.setattr(stage2_wiring, "_fit_group", fit_group)
    trajectory: list[list[np.ndarray]] = []
    call = full_data_fit._FullDataMeanField.__call__

    def traced(self, hyperparameters):
        points = call(self, hyperparameters)
        trajectory.append([np.array(self.elbo, copy=True), np.array(self.noise, copy=True)] + [
            np.array(values, copy=True) for hyper in hyperparameters for values in (hyper.coefficients, hyper.log_smoothing)
        ])
        return points

    monkeypatch.setattr(full_data_fit._FullDataMeanField, "__call__", traced)

    def fitted(name: str, switch: str | None):
        if switch is None:
            monkeypatch.delenv(training_codes.CODE_CACHE_SWITCH, raising=False)
        else:
            monkeypatch.setenv(training_codes.CODE_CACHE_SWITCH, switch)
        work_dir = tmp_path / name
        work_dir.mkdir()
        trajectory.clear()
        reads.clear()
        model = fit_model.fit(fit_model.FitRequest(
            store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=covariate[:, None], covariate_names=("c",),
            covariate_columns=np.ones((1, 1), dtype=bool), targets=np.where(~held_out, targets, np.nan)[:, None], training=~held_out[:, None],
            model_names=("trait/fold4",), trait_types=(TraitType.QUANTITATIVE,), research_ids=tuple(f"person{index}" for index in range(samples)),
            log_variance_offset=None, budget=_wiring_budget(), work_dir=work_dir, seed=5,
        ))
        save_model(tmp_path / f"{name}_model", model)
        return list(trajectory), _checkpoints(work_dir), _saved(tmp_path / f"{name}_model"), {method for when, method in reads if when == "stage2"}

    plain = fitted("plain", "0")
    assert plain[3] >= {"read_codes", "iter_codes"}
    kept = fitted("kept", None)
    assert not kept[3]
    assert len(kept[0]) == len(plain[0]) >= 2
    for kept_call, plain_call in zip(kept[0], plain[0]):
        assert len(kept_call) == len(plain_call)
        for left, right in zip(kept_call, plain_call):
            np.testing.assert_array_equal(left, right)
    assert plain[1] and sorted(kept[1]) == sorted(plain[1])
    for path, (metadata, arrays) in plain[1].items():
        assert kept[1][path][0] == metadata
        _same(kept[1][path][1], arrays)
    assert kept[2][0] == plain[2][0]
    _same(kept[2][1], plain[2][1])


def test_the_sharded_fill_runs_every_device_s_read_at_once_on_its_own_thread() -> None:
    """``ShardedDualSource.fill`` runs each shard's ``fill`` on a thread of its own, all at once (each waits at a
    barrier only the other can lift), says whether every shard holds its codes, and raises a shard's failure."""
    from sv_pgs.device_plan import ShardedDualSource, plan_blocks

    barrier = threading.Barrier(2, timeout=60)

    class Shard:
        def __init__(self, held: bool, failure: Exception | None = None) -> None:
            self.held, self.failure, self.thread = held, failure, None

        def fill(self) -> bool:
            self.thread = threading.current_thread().name
            barrier.wait()
            if self.failure is not None:
                raise self.failure
            return self.held

    def sharded(shards: list[Shard]) -> ShardedDualSource:
        return ShardedDualSource(shards, plan_blocks((0, 1), [3, 3], 10), [(0, 3), (3, 6)], 10, np)

    shards = [Shard(True), Shard(True)]
    assert sharded(shards).fill() and [shard.thread for shard in shards] == ["stage2-fill-0", "stage2-fill-1"]
    barrier.reset()
    assert not sharded([Shard(True), Shard(False)]).fill()
    barrier.reset()
    with pytest.raises(RuntimeError, match="device 1"):
        sharded([Shard(True), Shard(True, RuntimeError("device 1"))]).fill()


def test_without_a_device_fill_reads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the host, which holds no resident codes, ``fill`` reads nothing and says so."""
    _write_store(tmp_path / "store", _two_half_dosage(), "raw")
    store = DosageStore.open(tmp_path / "store")
    source = _source(store, "cpu", None, None)
    monkeypatch.setattr(store, "iter_codes", _no_read)
    assert not source.fill() and not source.resident_complete
    store.close()


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
@pytest.mark.parametrize("codec", ["raw", "zstd", "rowdict"])
@pytest.mark.parametrize("selected", [False, True])
def test_cuda_resident_codes_filled_from_kept_codes_are_the_store_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codec: str, selected: bool) -> None:
    """``fill`` from the kept codes reads nothing from the store and leaves the device's resident codes byte for byte
    the store read's (padding included), so every later read's products are the same bits."""
    _write_store(tmp_path / "store", _two_half_dosage(), codec)
    store = DosageStore.open(tmp_path / "store")
    sample_indices = np.arange(0, store.n_samples, 2, dtype=np.int64) if selected else None
    from_store = _source(store, "cuda", sample_indices, None)
    assert from_store.fill() and from_store.resident_complete
    expected = _products(from_store, 6)
    columns = np.arange(store.n_samples, dtype=np.int64) if sample_indices is None else sample_indices
    source = _source(store, "cuda", sample_indices, _kept(store, columns))
    monkeypatch.setattr(store, "iter_codes", _no_read)
    monkeypatch.setattr(store, "read_codes", _no_read)
    monkeypatch.setattr(store, "read_codes_to_device", _no_read)
    assert source.fill() and source.resident_complete
    for kept, read in zip(source._resident, from_store._resident, strict=True):
        assert np.array_equal(cupy.asnumpy(kept), cupy.asnumpy(read))
    assert _identical(_products(source, 6), expected)
    store.close()


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
def test_cuda_codes_evicted_mid_fill_leave_the_blocks_left_to_the_store(tmp_path: Path) -> None:
    """On a device too, codes evicted between two blocks of the fill leave the blocks left to the store: the resident
    set is still filled whole, and the image of a later read is the store-filled set's."""
    _write_store(tmp_path / "store", _two_half_dosage(), "zstd")
    store = DosageStore.open(tmp_path / "store")
    rng = np.random.default_rng(7)
    rights = [cupy.asarray(rng.standard_normal((block.shape[0], 2))) for block in BLOCK_ROWS]

    def image(source: StoreGenotypeBlockSource, evict=None) -> np.ndarray:
        total = cupy.zeros((source.sample_count, 2))
        for block_index, tile in source.iter_tiles():
            tile.accumulate_matmat(rights[block_index], total, FLOAT64_ROUNDING)
            if block_index == 0 and evict is not None:
                evict()
        return cupy.asnumpy(total)

    expected = image(_source(store, "cuda", None, None))
    with memory_scope(_budget("cuda")) as broker:
        codes = _kept(store, np.arange(store.n_samples, dtype=np.int64))
        source = _source(store, "cuda", None, codes)
        first = image(source, lambda: broker.evict_idle(HOST))
        assert codes.off is not None and source.resident_complete
    assert np.array_equal(_bits(first), _bits(expected))
    assert np.array_equal(_bits(image(source)), _bits(expected))
    store.close()


@pytest.mark.skipif(cupy is None or cupy.cuda.runtime.getDeviceCount() < 2, reason="needs two CUDA devices")
def test_cuda_the_sharded_fill_holds_every_device_s_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On two devices, ``stage2_source`` from the kept codes and its ``fill`` leave every shard's resident codes whole
    without a read of the store, and every block's products are the store-read source's."""
    from sv_pgs.compute_budget import ComputeBudget
    from sv_pgs.stage2_wiring import stage2_source

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    training = np.arange(_TRAINING, dtype=np.int64)
    candidates = np.arange(store.n_variants, dtype=np.int64)
    codes = TrainingCodes(candidates, training)
    statistics = compute_genotype_statistics(
        CodeTee(store, candidates, [codes]), training, np.column_stack([np.ones(_TRAINING), covariate[training]]), targets[training, None],
        ModelConfig(), _fit_budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    budget = ComputeBudget(
        device_kind="cuda", device_ids=(0, 1), device_names=("test", "test"), device_bytes=(1 << 33, 1 << 33),
        device_compute_capabilities=((8, 0), (8, 0)), host_bytes=1 << 31, cpu_threads=2,
    )
    from_store = stage2_source(store, statistics, budget, WORKSPACE_BYTES, training)
    assert from_store.fill()
    generator = np.random.default_rng(8)
    right = generator.standard_normal((from_store.variant_count, 2))
    expected = [cupy.asnumpy(tile.matmat(cupy.asarray(right[start:stop]))) for start, stop, tile in from_store.blocks()]
    source = stage2_source(store, statistics, budget, WORKSPACE_BYTES, training, codes)
    monkeypatch.setattr(store, "iter_codes", _no_read)
    monkeypatch.setattr(store, "read_codes", _no_read)
    monkeypatch.setattr(store, "read_codes_to_device", _no_read)
    assert source.fill()
    found = [cupy.asnumpy(tile.matmat(cupy.asarray(right[start:stop]))) for start, stop, tile in source.blocks()]
    for left, read in zip(found, expected, strict=True):
        assert np.array_equal(_bits(left), _bits(read))


@pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
def test_cuda_a_fresh_fill_keeps_its_restart_units_from_the_kept_codes_and_a_restart_fills_from_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One path for a device's codes: a fresh fit's fill takes them from Stage 0's kept codes and keeps each block as its
    restart unit (``_iter_prepared``), with no store read; a restarted fit, whose Stage 0 is its unit's and keeps no codes,
    fills from those units, again with no store read; and both resident sets are a store fill's, byte for byte."""
    _write_store(tmp_path / "store", _two_half_dosage(), "zstd")
    store = DosageStore.open(tmp_path / "store")
    from_store = _source(store, "cuda", None, None)
    assert from_store.fill()
    cache = CodeCache(RestartStore(tmp_path / "restart"), "model0", "store-digest")
    codes = _kept(store, np.arange(store.n_samples, dtype=np.int64))
    monkeypatch.setattr(store, "iter_codes", _no_read)
    monkeypatch.setattr(store, "read_codes", _no_read)
    monkeypatch.setattr(store, "read_codes_to_device", _no_read)
    fresh = _source(store, "cuda", None, codes, cache)
    assert fresh.fill() and fresh._origin == "stage0"
    assert len(cache.store.units("codes")) == len(BLOCK_ROWS)
    restarted = _source(store, "cuda", None, None, cache)
    assert restarted.fill() and restarted._origin == "units"
    for source in (fresh, restarted):
        for held, read in zip(source._resident, from_store._resident, strict=True):
            assert np.array_equal(cupy.asnumpy(held), cupy.asnumpy(read))
    store.close()

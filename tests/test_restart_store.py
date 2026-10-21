"""The restart store: a preempted fit resumes from what it made, and never from a unit its inputs or producer outgrew.

Units are whole or absent; a key holds exactly the inputs and the producing source; an engine edit keeps Stage 0, the
band and the codes, and a producer edit never reuses them; an EB start is an input-bound q solved and certified again,
never a completed fit. All fixtures are small synthetic CPU problems.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sv_pgs import full_data_fit, restart_store
from sv_pgs.binary_likelihood import BernoulliSites
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.full_data_fit import _MODEL_STATE_ARRAYS, _ModelState, block_grams, fit_full_data, stage0_lattice
from sv_pgs.genotype_statistics import DosageStoreTileSource, compute_genotype_statistics
from sv_pgs.restart_store import (
    REQUEST, RestartStore, WarmStart, definition_sources, facts, load_statistics, load_warm_start, module_sources, save_statistics,
    save_warm_start, state_key, unit_key,
)
from sv_pgs.scale_mixture_ep import AnnotationGroup, initial_hyperparameters, scale_mixture_prior, without_annotations
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from tests.test_dosage_store import _write_store


def _budget() -> ComputeBudget:
    return ComputeBudget(
        device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 31, cpu_threads=2
    )


def _sealed(store: RestartStore, kind: str, inputs: str, group: str, payload: bytes = b"payload") -> Path:
    named = facts({"inputs": inputs})
    with store.build(kind, unit_key(named), group=group, named=named, values={"block_cap": 64}) as directory:
        (directory / "ld").mkdir()
        (directory / "ld" / "gram.f32").write_bytes(payload)
        (directory / "statistics.npz").write_bytes(b"x")
    found = store.unit(kind, unit_key(named))
    assert found is not None
    return found[0]


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_a_sealed_unit_is_whole_or_absent(tmp_path: Path) -> None:
    store = RestartStore(tmp_path / "restart")
    path = _sealed(store, "stage0", "a", "model0")
    record = json.loads((path / restart_store.MANIFEST).read_text())
    assert record["files"] == {"ld/gram.f32": len(b"payload"), "statistics.npz": 1} and record["values"] == {"block_cap": 64}
    # Made again under its key, the whole unit stays: a unit is immutable.
    _sealed(store, "stage0", "a", "model0", b"another")
    assert (path / "ld" / "gram.f32").read_bytes() == b"payload" and json.loads((path / restart_store.MANIFEST).read_text())["id"] == record["id"]
    # A build that fails (a preemption is the same, less its cleanup) leaves no unit and no work.
    named = facts({"inputs": "b"})
    with pytest.raises(RuntimeError, match="preempted"):
        with store.build("stage0", unit_key(named), group="model0", named=named) as directory:
            (directory / "statistics.npz").write_bytes(b"half")
            raise RuntimeError("preempted")
    assert store.unit("stage0", unit_key(named)) is None
    assert [entry.name for entry in (tmp_path / "restart" / "stage0").iterdir()] == [path.name]
    # A file at another size, a missing manifest or a manifest of another key is no unit.
    (path / "ld" / "gram.f32").write_bytes(b"short")
    assert store.unit("stage0", path.name) is None and store.units("stage0") == []
    (path / "ld" / "gram.f32").write_bytes(b"payload")
    assert store.unit("stage0", path.name) is not None
    renamed = path.with_name("0" * 64)
    path.rename(renamed)
    assert store.unit("stage0", renamed.name) is None
    (renamed / restart_store.MANIFEST).unlink()
    assert store.units("stage0") == []


def test_superseded_released_and_unfinished_units_are_removed(tmp_path: Path) -> None:
    store = RestartStore(tmp_path / "restart")
    first = _sealed(store, "stage0", "a", "model0")
    second = _sealed(store, "stage0", "b", "model0")
    # A newer unit of a group's kind supersedes the group's older one; another group's stays.
    assert not first.exists() and second.exists()
    other = _sealed(store, "band", "c", "model3")
    checkpoints = store.mutable("checkpoints", "key", group=REQUEST, named={"source": "engine"})
    assert store.mutable("checkpoints", "key", group=REQUEST, named={"source": "engine"}) == checkpoints
    (checkpoints / "model0.npz").write_bytes(b"checkpoint")
    # A stop between a unit's making and its predecessor's release leaves both: asking for the unit again releases it.
    stale = store.mutable("checkpoints", "old", group=REQUEST, named={"source": "old engine"})
    assert store.mutable("checkpoints", "key", group=REQUEST, named={"source": "engine"}, supersede=True) == checkpoints and not stale.exists()
    store.release("model0")
    assert not second.exists() and other.exists()
    # Work a stopped process left: an unfinished unit and an archive's temporary.
    (tmp_path / "restart" / "band" / ".unfinished").mkdir()
    (checkpoints / ".model1.npz.tmp").write_bytes(b"half")
    store.clean()
    assert not (tmp_path / "restart" / "band" / ".unfinished").exists() and not (checkpoints / ".model1.npz.tmp").exists()
    # The request's completed models outlive every group's release.
    store.release()
    assert not other.exists() and (checkpoints / "model0.npz").read_bytes() == b"checkpoint"
    assert [path.name for path, _record in store.units("checkpoints")] == ["key"]


def test_a_unit_made_again_says_which_named_input_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(restart_store, "log", messages.append)
    store = RestartStore(tmp_path / "restart")
    named = facts({"store": "contents", "targets": np.arange(4.0), "source": "producer"})
    store.explain("stage0", "model0", named)
    assert messages == ["restart store: stage0 of model0: no earlier unit; making it"]
    with store.build("stage0", unit_key(named), group="model0", named=named):
        pass
    changed = facts({"store": "contents", "targets": np.arange(4.0) + 1.0, "source": "edited producer"})
    store.explain("stage0", "model0", changed)
    assert messages[-1].startswith("restart store: stage0 of model0 made again: source, targets changed since unit")


# ---------------------------------------------------------------------------
# Keys: the producing source and nothing else
# ---------------------------------------------------------------------------


def _edited(monkeypatch: pytest.MonkeyPatch, module: str, old: str, new: str) -> None:
    """``restart_store`` reads ``module``'s source with ``old`` replaced (once) by ``new``."""
    original = inspect.getsource

    def getsource(target):
        text = original(target)
        if getattr(target, "__name__", None) == module:
            assert text.count(old) == 1, (module, old)
            return text.replace(old, new)
        return text

    monkeypatch.setattr(restart_store, "inspect", SimpleNamespace(getsource=getsource))


_BAND = [("sv_pgs.gram_space", "summary_band"), ("sv_pgs.gram_space", "BandStore")]
_CODES = [("sv_pgs.store_block_source", "StoreGenotypeBlockSource")]
_STAGE0 = ("sv_pgs.genotype_statistics", "sv_pgs.dosage_store")


@pytest.mark.parametrize(
    ("module", "old", "new", "band", "codes", "stage0", "engine"),
    [
        # The sweeps beside the band builder in its module.
        ("sv_pgs.gram_space", "def _gram_sweep(", "def _gram_sweep_edited(", False, False, False, True),
        # The band builder itself, and a definition of another module it reaches.
        ("sv_pgs.gram_space", "byte for byte.", "byte for byte, edited.", True, False, False, True),
        ("sv_pgs.marginal_variances", "def extent_from_profile(", "def extent_from_profile( ", True, False, False, True),
        # The lag profiles Stage 0 sums as it projects each block (the band's key moves with the Stage 0 build it names).
        ("sv_pgs.lag_profiles", "def lag_excess_rows(", "def lag_excess_rows( ", False, False, True, True),
        # The engine: its mean-field oracle and its outer loop.
        ("sv_pgs.full_data_fit", "class _FullDataMeanField:", "class _FullDataMeanField:  # edited", False, False, False, True),
        ("sv_pgs.scale_mixture_ep", "def fit_hyperparameters(", "def fit_hyperparameters( ", False, False, False, True),
        # Stage 0 and the store's reads, which every product after it reads.
        ("sv_pgs.genotype_statistics", "def compute_genotype_statistics(", "def compute_genotype_statistics( ", False, False, True, True),
        ("sv_pgs.dosage_store", "class DosageStore:", "class DosageStore:  # edited", False, True, True, True),
        # The steps and the mirror around the fit.
        ("sv_pgs.run_mirror", "def push(", "def push( ", False, False, False, False),
        ("sv_pgs.workspace_pipeline", "def _fit_step(", "def _fit_step( ", False, False, False, False),
    ],
)
def test_each_product_is_keyed_by_its_producer_and_not_by_its_neighbours(
    monkeypatch: pytest.MonkeyPatch, module: str, old: str, new: str, band: bool, codes: bool, stage0: bool, engine: bool,
) -> None:
    """The band's key holds its builder and the definitions it reaches (``definition_sources``), across modules; the
    codes' key their class's; Stage 0's its modules and their imports; a completed model's key the engine's whole
    import closure (``fit_model.fit_identity``), which neither the steps nor the mirror are in. (The band's key also
    holds the Stage 0 unit's build, and the codes' key their rows, which an edit to Stage 0 can change.)"""
    import sv_pgs.run_mirror  # noqa: F401  (imported, so its source can be read)
    import sv_pgs.workspace_pipeline  # noqa: F401

    before = (definition_sources(_BAND), definition_sources(_CODES), module_sources(*_STAGE0), module_sources("sv_pgs.fit_model"))
    _edited(monkeypatch, module, old, new)
    after = (definition_sources(_BAND), definition_sources(_CODES), module_sources(*_STAGE0), module_sources("sv_pgs.fit_model"))
    assert [first != second for first, second in zip(before, after)] == [band, codes, stage0, engine]


def test_a_definition_a_key_names_must_exist() -> None:
    with pytest.raises(ValueError, match="defines no"):
        definition_sources([("sv_pgs.gram_space", "no_such_definition")])


# ---------------------------------------------------------------------------
# Stage 0's statistics
# ---------------------------------------------------------------------------


def test_stage0_is_found_by_its_inputs_and_serves_wherever_its_block_cap_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The block cap is a resource choice: a unit cut at a cap this machine's memory holds serves whatever the disk
    would choose now (a restored unit fills the disk the cap was read from), and a unit cut at a larger cap is made
    again at this machine's under its own key, which supersedes it."""
    from sv_pgs import stage2_wiring
    from tests.test_full_data_fit import _store

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    rows = np.arange(400)
    covariates = np.column_stack([np.ones(rows.shape[0]), covariate[rows]])
    restart = RestartStore(tmp_path / "restart")
    limits = {"memory": 2 * _BLOCK_CAP, "disk": 2 * _BLOCK_CAP}
    monkeypatch.setattr(stage2_wiring, "_cap_limits", lambda *args: (limits["memory"], store.n_variants))
    monkeypatch.setattr(stage2_wiring, "_block_cap", lambda *args: min(limits["memory"], limits["disk"]))
    made = []
    compute = stage2_wiring.compute_genotype_statistics

    def counted(*arguments):
        made.append(arguments[6])
        return compute(*arguments)

    monkeypatch.setattr(stage2_wiring, "compute_genotype_statistics", counted)

    def stage0():
        return stage2_wiring._stage0(
            store, np.arange(store.n_variants), rows, covariates, targets[rows, None], ModelConfig(minimum_minor_allele_frequency=0.0),
            _budget(), tmp_path, restart, "model0", "store",
        )

    first, _record = stage0()
    limits["disk"] = _BLOCK_CAP
    assert stage0()[0] == first and made == [2 * _BLOCK_CAP]
    limits["memory"] = _BLOCK_CAP
    second, record = stage0()
    assert made == [2 * _BLOCK_CAP, _BLOCK_CAP] and record["values"]["block_cap"] == _BLOCK_CAP and second != first
    assert [path for path, _record in restart.units("stage0")] == [second] and load_statistics(second).boundaries.block_cap == _BLOCK_CAP
    store.close()


def test_stage0_statistics_round_trip_with_their_ld_store(tmp_path: Path) -> None:
    from tests.test_genotype_statistics import BLOCK_CAP, _cpu_budget, _statistics_dataset

    source, training, covariates, targets = _statistics_dataset(21)
    expected = compute_genotype_statistics(source, training, covariates, targets, ModelConfig(), _cpu_budget(), BLOCK_CAP, tmp_path / "ld")
    save_statistics(tmp_path, expected)
    with np.load(tmp_path / restart_store.STATISTICS, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    found = load_statistics(tmp_path)
    for field in dataclasses.fields(expected):
        if field.name not in {"tie_map", "boundaries", "ld"}:
            np.testing.assert_array_equal(getattr(expected, field.name), getattr(found, field.name))
            assert type(getattr(expected, field.name)) is type(getattr(found, field.name)) or isinstance(getattr(found, field.name), np.ndarray)
    for name in ("kept_indices", "original_to_reduced", "signs"):
        np.testing.assert_array_equal(getattr(expected.tie_map, name), getattr(found.tie_map, name))
    assert found.boundaries.chromosomes == expected.boundaries.chromosomes and found.boundaries.block_cap == expected.boundaries.block_cap
    for chromosome in expected.boundaries.chromosomes:
        np.testing.assert_array_equal(expected.boundaries.cut_costs[chromosome], found.boundaries.cut_costs[chromosome])
    for block in range(found.ld.block_count):
        np.testing.assert_array_equal(expected.ld.read_gram(block), found.ld.read_gram(block))
        if found.ld.has_adjacent(block):
            np.testing.assert_array_equal(expected.ld.read_adjacent(block), found.ld.read_adjacent(block))


# ---------------------------------------------------------------------------
# Empirical Bayes warm starts
# ---------------------------------------------------------------------------


def _prior():
    return scale_mixture_prior(
        class_index=np.zeros(4, dtype=np.int64), log_variance_offset=np.zeros(4), annotation_design=np.zeros((4, 0)),
        annotation_groups=(), nodes=np.linspace(-4.0, 1.0, 7), floor=-5.0, top=2.0,
    )


def _state(binary=False):
    sites = BernoulliSites(np.arange(8) % 2, np.ones(8, dtype=bool), np.arange(8) / 10.0) if binary else None
    return _ModelState(
        mean=np.arange(4, dtype=np.float64), variance=np.ones(4), shift=np.arange(4, dtype=np.float64) / 3.0,
        third=np.zeros(4), fourth=np.full(4, 3.0), residual=np.arange(8, dtype=np.float64),
        noise=1.0, effective=2.0, mean_move=0.01, noise_gain=0.001, elbo=-12.5, version=7, sites=sites,
    )


def _equal(left, right):
    for field in dataclasses.fields(left):
        first, second = getattr(left, field.name), getattr(right, field.name)
        if dataclasses.is_dataclass(first):
            _equal(first, second)
        else:
            np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("binary", [False, True])
def test_a_warm_start_round_trips_without_a_pickle(tmp_path: Path, binary: bool) -> None:
    prior, state = _prior(), _state(binary)
    hyperparameters = initial_hyperparameters(prior)
    # An infinite smoothing weight is an actual edge of the prior, not a corrupt value.
    hyperparameters = dataclasses.replace(hyperparameters, log_smoothing=np.full_like(hyperparameters.log_smoothing, np.inf))
    key = state_key(store="contents", prior=prior)
    path = tmp_path / "model0.npz"
    save_warm_start(path, key, hyperparameters, state)
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    loaded_hyperparameters, loaded_state = load_warm_start(path, key, prior, 8, state.sites)
    _equal(hyperparameters, loaded_hyperparameters)
    _equal(state, loaded_state)
    if binary:
        assert loaded_state.sites.key == state.sites.key
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize("failure", ["key", "format", "truncated", "checksum", "shape", "nonfinite", "likelihood"])
def test_a_mismatched_or_corrupt_warm_start_is_ignored_with_a_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    prior, state = _prior(), _state()
    path = tmp_path / "model0.npz"
    messages: list[str] = []
    monkeypatch.setattr(restart_store, "log", messages.append)
    if failure == "shape":
        state = dataclasses.replace(state, residual=np.zeros(7))
    if failure == "nonfinite":
        state = dataclasses.replace(state, mean=np.full(4, np.nan))
    save_warm_start(path, "key", initial_hyperparameters(prior), state)
    if failure == "truncated":
        path.write_bytes(path.read_bytes()[:20])
    elif failure in ("format", "checksum"):
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.array(archive[name]) for name in archive.files}
        metadata = json.loads(str(values["metadata"]))
        metadata["format" if failure == "format" else "digest"] = "other"
        values["metadata"] = np.array(json.dumps(metadata))
        np.savez(path, **values)
    loaded = load_warm_start(path, "other" if failure == "key" else "key", prior, 8, _state(True).sites if failure == "likelihood" else None)
    assert loaded is None
    assert any("ignored model0.npz" in message for message in messages)


def test_a_failed_warm_start_write_keeps_the_previous_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prior, state = _prior(), _state()
    path = tmp_path / "model0.npz"
    save_warm_start(path, "key", initial_hyperparameters(prior), state)
    before = path.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError("preempted during the archive's write")

    monkeypatch.setattr(np, "savez", interrupted)
    with pytest.raises(OSError, match="preempted"):
        save_warm_start(path, "key", initial_hyperparameters(prior), dataclasses.replace(state, elbo=-10.0))
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize("part", ["store", "columns", "training", "targets", "covariates", "prior"])
def test_a_warm_start_key_changes_with_each_part_of_the_fitting_problem(part: str) -> None:
    parts = dict(store="content-a", columns=np.arange(8), training=np.ones(8, dtype=bool), targets=np.arange(8.0),
                 covariates=np.ones((8, 1)), prior=_prior())
    original = state_key(**parts)
    if part == "store":
        parts[part] = "content-b"
    elif part == "prior":
        parts[part] = dataclasses.replace(parts[part], log_variance_grid=parts[part].log_variance_grid + 0.1)
    else:
        parts[part] = np.array(parts[part], copy=True)
        parts[part].flat[0] = 0 if part in ("training", "covariates") else 1
    assert state_key(**parts) != original


_BLOCK_CAP = 64
"""Stage 0's least block cap (``genotype_statistics``), which holds the four columns in one block."""
_ROUNDING = 1e-10
"""A residual kept through a solve's updates against one made again from its means: float64 rounding over a few
hundred updates of 64 samples, far below this."""


def _problem(tmp_path: Path, model_count: int = 1):
    """Four exactly orthogonal, balanced dosage columns and an orthogonal residual (its fifth bit): no near ties or
    other coordinate-ascent modes are needed to test the restart."""
    from sv_pgs.dosage_store import DosageStore

    samples = 64
    rows = np.arange(samples)
    dosage = 2 * ((rows[:, None] >> np.arange(4)) & 1)
    design = dosage - 1.0
    target = design @ np.array([0.7, -0.4, 0.2, 0.0]) + 2 * ((rows >> 4) & 1) - 1.0
    targets = np.column_stack([target * (1.0 + 0.1 * model) for model in range(model_count)])
    covariates = np.ones((samples, 1))
    _write_store(tmp_path / "store", [{"chr22": (dosage.T * 1000).astype(np.int64)}])
    store = DosageStore.open(tmp_path / "store")
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, np.arange(4)), rows, covariates, targets, ModelConfig(), _budget(), _BLOCK_CAP, tmp_path / "ld",
    )
    draws, working_bytes = 16, 1 << 22
    noise = float(np.var(target, ddof=1))
    nodes, floor, top = stage0_lattice(statistics, 0, noise, np.zeros(4), 0.5 / draws)
    prior = scale_mixture_prior(
        class_index=np.zeros(4, dtype=np.int64), log_variance_offset=np.zeros(4), annotation_design=np.zeros((4, 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )

    def arguments():
        source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, _budget(), working_bytes))
        gaussian = DualGaussian(
            source=source, training=np.ones((samples, model_count)), targets=targets, offsets=np.zeros_like(targets),
            covariates=covariates, grams=block_grams(statistics), probe_count=0, seed=11,
        )
        return dict(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=draws, working_bytes=working_bytes,
                    seed=13, inference="mean_field")

    return arguments, store


def _warm(tmp_path: Path) -> WarmStart:
    return WarmStart(RestartStore(tmp_path / "restart"), "model0", "request")


def _solver(oracle):
    """The mean-field oracle the outer loop is handed inside its timer (``phase_timer.TimedOracle``)."""
    return getattr(oracle, "_oracle", oracle)


def test_an_interrupted_eb_resumes_at_its_saved_state_without_a_cold_first_solve_and_certifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments, store = _problem(tmp_path)
    original_outer = full_data_fit.fit_hyperparameters

    def interrupted(prior, starts, oracle, *args):
        assert all(point is not None for point in oracle(starts)), oracle.refusals
        raise RuntimeError("preempted before its curvature")

    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", interrupted)
    with pytest.raises(RuntimeError, match="preempted"):
        fit_full_data(**arguments(), warm_start=_warm(tmp_path))
    (path,) = (tmp_path / "restart" / "eb").glob("*/model0.npz")
    saved_hyperparameters, saved_state = load_warm_start(path, path.parent.name, arguments()["prior"], 64, None)
    original_solve = full_data_fit._FullDataMeanField._solve_model
    original_cold = full_data_fit._FullDataMeanField._cold_state
    first = []

    def checked_solve(self, model, hyperparameters, **kwargs):
        if not first:
            first.append(self)
            assert self.refreshes == 0
            _equal(saved_hyperparameters, hyperparameters)
            for name in _MODEL_STATE_ARRAYS[:-1]:
                np.testing.assert_array_equal(getattr(self, name)[:, model], getattr(saved_state, name))
            # The residual is made again from the saved means by this engine's read: equal to the kept one but for
            # the kept one's accumulated rounding.
            np.testing.assert_allclose(self.residual[model], saved_state.residual, rtol=0.0, atol=_ROUNDING * np.abs(saved_state.residual).max())
        return original_solve(self, model, hyperparameters, **kwargs)

    def checked_cold(self, model):
        assert not first or self is not first[0] or self.refreshes != 0, "ran a cold competitor on the resumed first call"
        return original_cold(self, model)

    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", original_outer)
    monkeypatch.setattr(full_data_fit._FullDataMeanField, "_solve_model", checked_solve)
    monkeypatch.setattr(full_data_fit._FullDataMeanField, "_cold_state", checked_cold)
    resumed = fit_full_data(**arguments(), warm_start=_warm(tmp_path))
    certificate = resumed.certificate
    assert first
    assert np.all(certificate.outer_criterion_met), certificate.refusals
    assert np.all(certificate.remaining_gain <= 0.5 / 16)
    assert np.all(certificate.mean_move <= certificate.draw_tolerance)
    assert np.all(certificate.noise_gain <= 0.5 / 16)
    assert np.all(certificate.budget_unresolved == 0)
    store.close()


def test_a_partly_saved_group_restores_only_the_model_whose_start_is_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments, store = _problem(tmp_path, model_count=2)
    original_save = full_data_fit.save_warm_start

    def save_first(path, key, hyperparameters, state):
        original_save(path, key, hyperparameters, state)
        raise RuntimeError("preempted between models")

    monkeypatch.setattr(full_data_fit, "save_warm_start", save_first)
    with pytest.raises(RuntimeError, match="between models"):
        fit_full_data(**arguments(), warm_start=_warm(tmp_path))
    (path,) = (tmp_path / "restart" / "eb").glob("*/model*.npz")
    assert path.name == "model0.npz"
    saved_hyperparameters, saved_state = load_warm_start(path, path.parent.name, arguments()["prior"], 64, None)
    inspected = []

    def inspect_starts(prior, starts, oracle, *args):
        oracle = _solver(oracle)
        _equal(starts[0], saved_hyperparameters)
        np.testing.assert_array_equal(oracle.mean[:, 0], saved_state.mean)
        np.testing.assert_allclose(oracle.residual[0], saved_state.residual, rtol=0.0, atol=_ROUNDING * np.abs(saved_state.residual).max())
        np.testing.assert_array_equal(oracle.mean[:, 1], 0.0)
        assert oracle._at[0] is not None and oracle._at[1] is None
        inspected.append(True)
        raise RuntimeError("inspected the warm and the cold start")

    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", inspect_starts)
    with pytest.raises(RuntimeError, match="inspected"):
        fit_full_data(**arguments(), warm_start=_warm(tmp_path))
    assert inspected
    store.close()


def test_the_nested_unannotated_solve_resumes_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments, store = _problem(tmp_path)
    plain_prior = arguments()["prior"]
    prior = scale_mixture_prior(
        class_index=plain_prior.class_index, log_variance_offset=plain_prior.log_variance_offset,
        annotation_design=np.arange(4.0)[:, None], annotation_groups=(AnnotationGroup(np.array([0]), np.eye(1), "synthetic"),),
        nodes=plain_prior.log_variance_grid, floor=plain_prior.kernel_floor, top=plain_prior.kernel_top,
    )

    def interrupted(model_prior, starts, oracle, *args):
        assert not model_prior.annotation_groups
        assert all(point is not None for point in oracle(starts)), oracle.refusals
        raise RuntimeError("preempted in the nested base fit")

    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", interrupted)
    with pytest.raises(RuntimeError, match="nested base"):
        fit_full_data(**(arguments() | {"prior": prior}), warm_start=_warm(tmp_path))
    (path,) = (tmp_path / "restart" / "eb").glob("*/model0.npz")
    saved_hyperparameters, saved_state = load_warm_start(path, path.parent.name, without_annotations(prior), 64, None)
    inspected = []

    def inspect_starts(model_prior, starts, oracle, *args):
        oracle = _solver(oracle)
        assert not model_prior.annotation_groups
        _equal(starts[0], saved_hyperparameters)
        np.testing.assert_array_equal(oracle.mean[:, 0], saved_state.mean)
        inspected.append(True)
        raise RuntimeError("inspected the nested start")

    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", inspect_starts)
    with pytest.raises(RuntimeError, match="inspected the nested"):
        fit_full_data(**(arguments() | {"prior": prior}), warm_start=_warm(tmp_path))
    assert inspected
    store.close()


def test_keeping_warm_starts_does_not_change_an_uninterrupted_fit(tmp_path: Path) -> None:
    arguments, store = _problem(tmp_path)
    plain = fit_full_data(**arguments())
    kept = fit_full_data(**arguments(), warm_start=_warm(tmp_path))
    np.testing.assert_array_equal(plain.certificate.state_digest, kept.certificate.state_digest)
    np.testing.assert_array_equal(plain.certificate.outer_iterations, kept.certificate.outer_iterations)
    assert list((tmp_path / "restart" / "eb").glob("*/model0.npz"))
    store.close()


# ---------------------------------------------------------------------------
# End to end: a preempted group, restarted on another disk from the mirrored store
# ---------------------------------------------------------------------------


def test_a_group_preempted_mid_eb_resumes_on_a_fresh_disk_from_its_mirrored_restart_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The band route's group of one model (``stage2_wiring.fit_models``) is preempted at its oracle's third call; its
    restart store is pushed to a mirror and restored into a fresh run directory, where the fit resumes in a fresh work
    directory: Stage 0, the far field and the band are read, not made, the outer loop starts at the saved state, and the
    lattice check and the certificate run as on any fit. A third run finds the completed model checkpointed, and a fit
    whose engine changed is made again with the change logged."""
    from sv_pgs import stage2_wiring
    from sv_pgs.run_mirror import DirectoryRemote, push, restore
    from tests.test_full_data_fit import _store

    store, covariate, targets, _genetic = _store(tmp_path / "store", 7)
    samples = store.n_samples
    training = (np.arange(samples) < samples * 4 // 5)[:, None]
    monkeypatch.setattr(stage2_wiring, "pass_costs", lambda band, source, xp: (np.inf, 0.0))
    identity = facts({"source": "engine", "problem": "synthetic"})

    def fit(run: Path, work: str, named: dict[str, str] = identity):
        (run / work).mkdir(parents=True)
        return stage2_wiring.fit_models(
            store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=np.column_stack([np.ones(samples), covariate]),
            covariate_columns=np.ones((1, 2), dtype=bool), targets=np.where(training, targets[:, None], 0.0), training=training,
            trait_types=(TraitType.QUANTITATIVE,), log_variance_offset=None, budget=_budget(), work_dir=run / work, seed=5, draw_count=16,
            checkpoint_key=unit_key(named), restart=RestartStore(run / "restart" / "snv"), identity=named, store_key="store", request_key="request",
        )

    reference = fit(tmp_path / "reference", "work")
    original_call = full_data_fit._FullDataMeanField.__call__
    calls: list[int] = []

    def preempted(self, hyperparameters):
        points = original_call(self, hyperparameters)
        calls.append(len(calls))
        if len(calls) == 3:
            raise RuntimeError("preempted mid-EB")
        return points

    monkeypatch.setattr(full_data_fit._FullDataMeanField, "__call__", preempted)
    with pytest.raises(RuntimeError, match="preempted mid-EB"):
        fit(tmp_path / "first", "work")
    first = RestartStore(tmp_path / "first" / "restart" / "snv")
    assert [len(first.units(kind)) for kind in ("stage0", "farfield", "band")] == [1, 1, 1]
    (saved,) = first.directory.glob("eb/*/model0.npz")
    remote = DirectoryRemote(tmp_path / "mirror")
    push(tmp_path / "first", remote)
    restore(tmp_path / "second", remote)
    capsys.readouterr()

    monkeypatch.setattr(full_data_fit._FullDataMeanField, "__call__", original_call)
    for builder in ("compute_genotype_statistics", "summary_band", "far_field_scale"):
        monkeypatch.setattr(stage2_wiring, builder, lambda *args, builder=builder, **kwargs: pytest.fail(f"made {builder} again"))
    outer = full_data_fit.fit_hyperparameters
    started: list[bool] = []

    def resumed_outer(model_prior, starts, oracle, *arguments):
        if not started:
            solver = _solver(oracle)
            hyperparameters, state = load_warm_start(saved, saved.parent.name, model_prior, solver.residual[0].shape[0], None)
            _equal(hyperparameters, starts[0])
            np.testing.assert_array_equal(solver.mean[:, 0], state.mean)
            assert solver.noise[0] == state.noise
            started.append(True)
        return outer(model_prior, starts, oracle, *arguments)

    monkeypatch.setattr(full_data_fit, "fit_hyperparameters", resumed_outer)
    checked = []
    lattice_check = stage2_wiring.lattice_check

    def counted(*arguments):
        checked.append(True)
        return lattice_check(*arguments)

    monkeypatch.setattr(stage2_wiring, "lattice_check", counted)
    resumed = fit(tmp_path / "second", "fresh-work")
    messages = capsys.readouterr().err
    assert started and checked
    for message in ("Stage 0 reused from its restart unit", "far field: reused its restart unit", "summary band reused from its restart unit", "eb warm start: loaded"):
        assert message in messages
    certificate = resumed.certificate
    assert np.all(np.isfinite(certificate.remaining_gain) | ~np.asarray(certificate.outer_criterion_met))
    assert np.all(certificate.noise_gain <= 0.5 / 16)
    assert np.corrcoef(resumed.scoring[0].coefficients, reference.scoring[0].coefficients)[0, 1] > 0.99
    # The group is checkpointed: its Stage 0, far field, band and EB starts are released, its checkpoint kept.
    second = RestartStore(tmp_path / "second" / "restart" / "snv")
    assert [len(second.units(kind)) for kind in restart_store.KINDS] == [0, 0, 0, 0, 0, 1]

    monkeypatch.setattr(stage2_wiring, "_fit_group", lambda *args: pytest.fail("fitted a checkpointed group again"))
    again = fit(tmp_path / "second", "third-work")
    np.testing.assert_array_equal(again.scoring[0].coefficients, resumed.scoring[0].coefficients)

    def covariates_alone(_store, _columns, covariates, model_targets, _measurement, _budget, _work_dir, _seed, draw_count, models, *_rest):
        return [
            stage2_wiring._null_genetic_model(stage2_wiring._covariate_least_squares(covariates, model_targets[:, model]), draw_count, "stand-in")
            for model in range(model_targets.shape[1])
        ]

    monkeypatch.setattr(stage2_wiring, "_fit_group", covariates_alone)
    capsys.readouterr()
    edited = facts({"source": "edited engine", "problem": "synthetic"})
    fit(tmp_path / "second", "edited-work", edited)
    assert "restart store: checkpoints of request made again: source changed since unit" in capsys.readouterr().err
    assert [path.name for path, _record in second.units("checkpoints")] == [unit_key(edited)]
    store.close()

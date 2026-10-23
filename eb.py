"""Each mean-field EB solve keeps its models' best states as resume units, and starts from an earlier attempt's.

This is d30ea8a's ``fit_full_data(warm_start=...)`` attached from outside the staged tree, which cannot change:

- ``fit_model.fit_models`` (wrapped) names the request: ``fit_model.fit``'s checkpoint key (the engine's source, the
  store, the offsets, the problem and the prior form) and the arm (the fit step's work directory, work/<arm>). Once
  it returns, every model is checkpointed and the request's units are released.
- ``full_data_fit.fit_full_data`` (wrapped, where ``stage2_wiring`` calls it too) reads the solve's layout before any
  oracle reweights the dual solver: the request, the route (the band's or the samples'), Stage 0's rows, ties and
  blocks, the dual solver's training mask, targets, covariates and offsets, the binary sites, the draws and the seed.
- Each oracle that ``fit_full_data``'s own ``solve`` builds (``_FullDataMeanField`` or ``_GramMeanField``; never the
  mode mixture's components, nor EP's) belongs to one unit, keyed by the layout, the solve's prior and how many solves
  of that layout and prior this process began before it, so each of a group's solves (the nested base and annotated
  priors, the band fit's sample-route continuation, a moved lattice's refit) resumes its own. A model whose unit holds
  a whole state starts there: its hyperparameters and noise replace the start's before the oracle is built, and its
  moments and sites are installed after, its residual made again from the means by this engine's own read. Each
  improvement of a model's best ELBO is saved before the call's fixed point is formed (``_fixed_point``) and again
  when the call returns.
- Where every model of an annotated fit from the prior has a saved state on the annotated prior, the fit continues
  from those states (``starts``, ``start_noise``), as a lattice refit does: the nested search on the prior without
  annotations is behind it.

A process reads a unit at most once and never one it wrote, so a fit with no units on its disk takes exactly the path
it takes without this module. A resumed solve can take another path to its fixed point, which the outer loop solves and
certifies again; the outer loop's search history is not kept. Any failure here is logged and the fit goes on as
without it."""
from __future__ import annotations

import functools
import sys
import types
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.live_resume.units import load_state, release, save_state, state_key, unit_directory
from sv_pgs.progress import log


@dataclass(frozen=True)
class _Request:
    key: str
    arm: str


@dataclass
class _Call:
    """One ``fit_full_data`` call in progress: its request, layout, models' sites and residual length, and the saved
    states its continuation already read, by unit."""

    request: _Request
    layout: str
    sites: list
    residual_length: int
    peeked: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Unit:
    path: Path
    key: str


@dataclass
class _Sink:
    """Where an oracle keeps its models' best states, and the ELBO each model's saved state has."""

    unit: _Unit
    request: _Request
    saved: np.ndarray
    failed: bool = False


@dataclass
class _Hooks:
    root: Path | None = None
    solve_code: types.CodeType | None = None
    requests: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    occurrences: Counter = field(default_factory=Counter)
    written: set = field(default_factory=set)
    consumed: set = field(default_factory=set)
    originals: list = field(default_factory=list)


_HOOKS = _Hooks()


def _nested_code(function: Any, name: str) -> types.CodeType | None:
    """The code of ``function``'s nested function ``name`` (the frame that builds an oracle), or None."""
    found = [const for const in function.__code__.co_consts if isinstance(const, types.CodeType) and const.co_name == name]
    return found[0] if len(found) == 1 else None


def _unit(call: _Call, prior: Any, begin: bool) -> _Unit:
    """The unit of the next solve of ``call``'s layout on ``prior``; ``begin`` counts that solve as begun."""
    slot = (call.layout, state_key(prior=prior))
    occurrence = _HOOKS.occurrences[slot]
    if begin:
        _HOOKS.occurrences[slot] += 1
    key = state_key(layout=call.layout, prior=slot[1], occurrence=occurrence)
    assert _HOOKS.root is not None
    return _Unit(_HOOKS.root / call.request.arm / "eb" / key, key)


def _load(call: _Call, unit: _Unit, prior: Any) -> list:
    """Each model's saved (hyperparameters, state) in ``unit``, or None per model."""
    if not unit.path.is_dir():
        return [None] * len(call.sites)
    return [load_state(unit.path / f"model{model}.npz", unit.key, prior, call.residual_length, call.sites[model]) for model in range(len(call.sites))]


def _entries(call: _Call, unit: _Unit, prior: Any) -> list:
    """``unit``'s saved states for the oracle being built, once per process and never from a unit it wrote."""
    if unit.path in _HOOKS.written or unit.path in _HOOKS.consumed:
        return [None] * len(call.sites)
    _HOOKS.consumed.add(unit.path)
    if unit.path in call.peeked:
        return call.peeked.pop(unit.path)
    return _load(call, unit, prior)


def _flush(oracle: Any) -> None:
    """Save each model whose best ELBO rose since its last save (the oracle's ``best_state``, ``best_hyperparameters``)."""
    sink = oracle.__dict__.get("_live_resume")
    if sink is None or sink.failed:
        return
    for model in range(oracle.model_count):
        state = oracle.best_state[model]
        if state is None or not oracle.best_elbo[model] > sink.saved[model]:
            continue
        try:
            assert _HOOKS.root is not None
            directory = unit_directory(_HOOKS.root, sink.request.arm, sink.unit.key, sink.request.key)
            _HOOKS.written.add(sink.unit.path)
            save_state(directory / f"model{model}.npz", sink.unit.key, oracle.best_hyperparameters[model], state)
        except Exception as error:
            sink.failed = True
            log(f"live resume: this solve's states are no longer kept ({type(error).__name__}: {error})")
            return
        sink.saved[model] = oracle.best_elbo[model]


def _wrap_fit_models(original: Any) -> Any:
    @functools.wraps(original)
    def fit_models(*arguments: Any, **named: Any) -> Any:
        key = named.get("checkpoint_key")
        if arguments or key is None or "work_dir" not in named:
            return original(*arguments, **named)
        request = _Request(str(key), Path(named["work_dir"]).name)
        _HOOKS.requests.append(request)
        try:
            fitted = original(**named)
        finally:
            _HOOKS.requests.pop()
        try:
            assert _HOOKS.root is not None
            removed = release(_HOOKS.root, request.arm, request.key)
            if removed:
                log(f"live resume: {request.arm}'s models are checkpointed; released {removed} EB units")
        except Exception as error:
            log(f"live resume: {request.arm}'s EB units were not released ({type(error).__name__}: {error})")
        return fitted

    return fit_models


def _call(named: dict[str, Any]) -> _Call:
    from sv_pgs.dual_solve import _host
    from sv_pgs.tie_members import TieGroups

    gaussian, statistics, band = named["gaussian"], named["statistics"], named.get("band")
    sites = list(named["sites"]) if named.get("sites") is not None else [None] * gaussian.model_count
    training = np.asarray(_host(gaussian.training), dtype=np.float64)
    request = _HOOKS.requests[-1]
    layout = state_key(
        request=request.key, route="_GramMeanField" if band is not None else "_FullDataMeanField",
        rows=np.asarray(statistics.active_rows), ties=TieGroups.from_tie_map(statistics.tie_map), blocks=np.asarray(statistics.block_of_reduced),
        training=training, targets=np.asarray(_host(gaussian.targets)), covariates=np.asarray(_host(gaussian.covariates)),
        offsets=np.asarray(_host(gaussian.offsets)), sites=sites, draw_count=int(named["draw_count"]), seed=int(named["seed"]),
    )
    return _Call(request, layout, sites, int(band.group_count) if band is not None else int(training.shape[0]))


def _continued(call: _Call, named: dict[str, Any]) -> dict[str, Any]:
    """``fit_full_data``'s arguments, continued from the annotated prior's saved states where a fit from the prior has
    one for every model."""
    prior = named["prior"]
    if named.get("starts") is not None or not prior.annotation_groups:
        return named
    unit = _unit(call, prior, begin=False)
    if unit.path in _HOOKS.written or unit.path in _HOOKS.consumed:
        return named
    entries = _load(call, unit, prior)
    if any(entry is None for entry in entries):
        return named
    call.peeked[unit.path] = entries
    log("live resume: every model has a saved state on the annotated prior: the fit continues there; the nested search is behind it")
    return {**named, "starts": [entry[0] for entry in entries], "start_noise": np.array([entry[1].noise for entry in entries], dtype=np.float64)}


def _wrap_fit_full_data(original: Any) -> Any:
    @functools.wraps(original)
    def fit_full_data(*arguments: Any, **named: Any) -> Any:
        if arguments or not _HOOKS.requests or named.get("inference", "ep") != "mean_field":
            return original(*arguments, **named)
        try:
            call = _call(named)
            named = _continued(call, named)
        except Exception as error:
            log(f"live resume: off for this fit ({type(error).__name__}: {error})")
            return original(**named)
        _HOOKS.calls.append(call)
        try:
            return original(**named)
        finally:
            _HOOKS.calls.pop()

    return fit_full_data


def _wrap_init(original: Any, cls: type) -> Any:
    @functools.wraps(original)
    def __init__(self: Any, gaussian: Any, statistics: Any, prior: Any, draw_count: Any, working_bytes: Any, seed: Any, starts: Any, noise: Any,
                 *arguments: Any, **named: Any) -> None:
        call = _HOOKS.calls[-1] if _HOOKS.calls else None
        if call is None or type(self) is not cls or sys._getframe(1).f_code is not _HOOKS.solve_code:
            original(self, gaussian, statistics, prior, draw_count, working_bytes, seed, starts, noise, *arguments, **named)
            return
        unit: _Unit | None = None
        entries: list = [None] * len(call.sites)
        try:
            unit = _unit(call, prior, begin=True)
            entries = _entries(call, unit, prior)
        except Exception as error:
            log(f"live resume: no saved state read for this solve ({type(error).__name__}: {error})")
        if not isinstance(starts, list):
            entries = [None] * len(call.sites)
        if any(entry is not None for entry in entries):
            noise = np.array(noise, dtype=np.float64, copy=True)
            for model, entry in enumerate(entries):
                if entry is not None:
                    starts[model], noise[model] = entry[0], entry[1].noise
        original(self, gaussian, statistics, prior, draw_count, working_bytes, seed, starts, noise, *arguments, **named)
        for model, entry in enumerate(entries):
            if entry is None:
                continue
            try:
                # A solver version is this process's: the saved one names nothing here. The residual is made again from
                # the saved means by this engine's own read (``_residual_of``). The first call's solve starts here, not
                # at the cold start (``_solved``), as d30ea8a's warm start does.
                self._restore_state(self._charged(replace(entry[1], version=self.version)), model)
                self.residual[model] = self._residual_of(model)
                log(f"live resume: model {model} resumes its EB at its saved state (ELBO {entry[1].elbo:.6g}) from unit {unit.key[:16] if unit else ''}")
            except Exception as error:
                # The saved hyperparameters and noise are a start like any other; the moments go back to the cold start.
                self._cold_state(model)
                log(f"live resume: model {model}'s saved state was not installed ({type(error).__name__}: {error}); its EB starts cold at the saved hyperparameters")
        if unit is not None:
            self.__dict__["_live_resume"] = _Sink(unit, call.request, np.full(len(call.sites), -np.inf))

    return __init__


def _wrap_call(original: Any) -> Any:
    @functools.wraps(original)
    def __call__(self: Any, hyperparameters: Any) -> Any:
        points = original(self, hyperparameters)
        _flush(self)
        return points

    return __call__


def _wrap_fixed_point(original: Any) -> Any:
    @functools.wraps(original)
    def _fixed_point(self: Any, model: int, hyperparameters: Any) -> Any:
        _flush(self)
        return original(self, model, hyperparameters)

    return _fixed_point


def install(root: Path) -> bool:
    """Keep and resume EB states under ``root`` (the run directory's ``units.ROOT``); False where this engine is not the
    one the hooks were written against (nothing is then changed)."""
    from sv_pgs import fit_model, full_data_fit, stage2_wiring

    if _HOOKS.root is not None:
        _HOOKS.root = Path(root)
        return True
    oracle, gram = full_data_fit._FullDataMeanField, full_data_fit._GramMeanField
    solve_code = _nested_code(full_data_fit.fit_full_data, "solve")
    methods = ("_restore_state", "_charged", "_residual_of", "_cold_state", "_fixed_point", "_state")
    fields = ("best_state", "best_elbo", "best_hyperparameters", "residual", "version", "model_count")
    if (solve_code is None or any(not hasattr(oracle, name) for name in methods) or any(name not in oracle.__init__.__code__.co_names for name in fields)
            or "__call__" in gram.__dict__ or "_fixed_point" in gram.__dict__
            or fit_model.fit_models is not stage2_wiring.fit_models or stage2_wiring.fit_full_data is not full_data_fit.fit_full_data):
        log("live resume: this engine is not the one the EB hooks were written for; EB states are not kept")
        return False
    _HOOKS.root, _HOOKS.solve_code = Path(root), solve_code
    fitting, full = _wrap_fit_models(stage2_wiring.fit_models), _wrap_fit_full_data(full_data_fit.fit_full_data)
    replacements = [
        (fit_model, "fit_models", fitting), (stage2_wiring, "fit_models", fitting),
        (full_data_fit, "fit_full_data", full), (stage2_wiring, "fit_full_data", full),
        (oracle, "__init__", _wrap_init(oracle.__init__, oracle)), (gram, "__init__", _wrap_init(gram.__init__, gram)),
        (oracle, "__call__", _wrap_call(oracle.__call__)), (oracle, "_fixed_point", _wrap_fixed_point(oracle._fixed_point)),
    ]
    for owner, name, value in replacements:
        _HOOKS.originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)
    return True


def uninstall() -> None:
    """Put back every wrapped function and forget this process's units (tests)."""
    while _HOOKS.originals:
        owner, name, value = _HOOKS.originals.pop()
        setattr(owner, name, value)
    _HOOKS.__init__()

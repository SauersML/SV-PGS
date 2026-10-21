"""The fit's restart store: what a preempted fit resumes from, outside the fit step's keyed directory.

A whole-genome fit spends its first hours on products it can make again exactly: Stage 0's statistics and LD store
(43-50 minutes and 150-169 GB on All of Us whole-genome arms [real]), the far-field scale, the summary band (9-14
minutes, 82-98 GB [real]) and the training samples' signed genotype codes the devices hold (a 20-minute read of the
547 GB store [real]). Then empirical Bayes runs for hours. A Spot preemption took all of it, and a step key changed by
any edit (the fit step is keyed by its code) took the completed models too. This store keeps each of them as a unit
keyed by exactly what determines it, so a restarted fit makes only what it lacks:

- **sealed units** (``build``): ``<kind>/<key>/`` with every file closed before ``MANIFEST`` is written and the
  directory renamed into place, the manifest naming each file's size. Stage 0 (``stage0``: its arrays,
  ``save_statistics``, beside its LD files), the far-field scale (``farfield``), the band (``band``) and each LD
  block's codes (``codes``). A unit is immutable: one made again under its key replaces nothing that is whole.
- **mutable units** (``mutable``): ``<kind>/<key>/`` whose files each check themselves and are replaced atomically:
  each model's best mean-field start (``eb``: ``save_warm_start``) and each completed model's engine checkpoint
  (``checkpoints``: ``fit_checkpoint``).

A key is the digest of named parts (``facts``), each part's own digest recorded in the manifest, so a unit made again
says which part changed (``explain``). The parts are the inputs and the source of the code that produces the bytes,
never more: ``module_sources`` for a producer module and every package module it imports, ``definition_sources`` for
the producing definitions of a module that also holds other code (``gram_space``'s band builder beside its sweeps),
so an engine edit keeps Stage 0 and an edit to Stage 0 never reuses a stale one. Each unit belongs to a group (the
LD-sharing group of models it serves, or ``REQUEST`` for the request's completed models): a group's units are
released (``release``) once its models are checkpointed, and a newer unit of a kind supersedes the group's older ones.

The run mirror (``run_mirror``) carries the store between jobs: sealed units payload first and manifest last, and
never replaces a whole remote unit; the fit itself never touches the bucket, so publishing a unit is a local rename
and its upload overlaps the fit. No unit holds a participant-level value outside the run's own disk and mirror.
"""
from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import importlib
import inspect
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from sv_pgs.artifact import named_digest
from sv_pgs.fit_checkpoint import save_archive
from sv_pgs.progress import log

FORMAT = "svpgs-restart-unit-v1"
MANIFEST = "UNIT.json"
"""A unit's manifest, written last: its format, kind, key, group, facts, values, build id and (sealed) file sizes."""
REQUEST = "request"
"""The group of the request's completed models, which no group's release removes."""
KINDS = ("stage0", "farfield", "band", "codes", "eb", "checkpoints")
"""The units' kinds, each a directory of the store (the module docstring); nothing else in it is the store's."""
STATISTICS = "statistics.npz"
WARM_FORMAT = "svpgs-eb-warm-start-v1"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _part(value: Any) -> str:
    """One named part's digest: a string's bytes, a scalar's JSON, or an array's dtype, shape and bytes."""
    if isinstance(value, str):
        return hashlib.sha256(value.encode()).hexdigest()
    if value is None or isinstance(value, (bool, int, float)):
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()
    return named_digest({"value": np.asarray(value)})


def facts(parts: Mapping[str, Any]) -> dict[str, str]:
    """Each named part as its own digest, in name order (a float by its repr, which reads back exactly)."""
    return {name: _part(parts[name]) for name in sorted(parts)}


def unit_key(named: Mapping[str, str]) -> str:
    """The unit's key: the digest of its facts, names and values separated so neither can run into the next."""
    return hashlib.sha256(json.dumps(dict(named), sort_keys=True).encode()).hexdigest()


def module_sources(*roots: str) -> str:
    """The digest of the ``roots`` modules and every ``sv_pgs`` module they import, transitively (function-level
    imports included): the code a producer runs, as ``workspace_pipeline.module_closure`` keys a step. A relative
    import is refused rather than silently left out."""
    found: dict[str, str] = {}
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        source = inspect.getsource(importlib.import_module(name))
        found[name] = hashlib.sha256(source.encode()).hexdigest()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError(f"{name} uses a relative import; the restart store keys absolute sv_pgs imports only")
                names = [f"{node.module}.{alias.name}" for alias in node.names] if node.module == "sv_pgs" else [node.module or ""]
            else:
                continue
            pending.extend(imported for imported in names if imported.startswith("sv_pgs."))
    return hashlib.sha256(json.dumps(dict(sorted(found.items()))).encode()).hexdigest()


def _definitions(module: str) -> tuple[str, dict[str, list[ast.stmt]], dict[str, tuple[str, str | None]]]:
    """A module's source, its top-level definitions by name (functions, classes, assignments), and the names its
    top-level ``from sv_pgs...`` imports bind, each as (module, name) (name None for a module imported whole)."""
    source = inspect.getsource(importlib.import_module(module))
    definitions: dict[str, list[ast.stmt]] = {}
    imports: dict[str, tuple[str, str | None]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for item in ast.walk(target):
                    if isinstance(item, ast.Name):
                        definitions.setdefault(item.id, []).append(node)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None and node.module.startswith("sv_pgs"):
            for alias in node.names:
                if node.module == "sv_pgs":
                    imports[alias.asname or alias.name] = (f"sv_pgs.{alias.name}", None)
                else:
                    imports[alias.asname or alias.name] = (node.module, alias.name)
    return source, definitions, imports


def definition_sources(roots: Iterable[tuple[str, str]]) -> str:
    """The digest of the source of the definitions ``roots`` ((module, name) pairs) and of every definition they name,
    followed across ``sv_pgs`` modules through each module's imports (top-level and inside the definitions): the code a
    producer runs where its module also holds code it never runs (``gram_space``'s sweeps beside the band builder).
    A module imported whole and named in a definition is followed whole. A method reached through an object of another
    module's class is followed only where that class is named; the units built here name the Stage 0 unit whose
    classes they read, whose key holds Stage 0's whole modules."""
    cache: dict[str, tuple[str, dict[str, list[ast.stmt]], dict[str, tuple[str, str | None]]]] = {}
    chosen: dict[tuple[str, str], str] = {}
    pending = list(roots)
    while pending:
        module, name = pending.pop()
        if (module, name) in chosen:
            continue
        if module not in cache:
            cache[module] = _definitions(module)
        source, definitions, imports = cache[module]
        if name == "*":
            nodes: list[ast.stmt] = [node for entries in definitions.values() for node in entries]
            chosen[(module, name)] = source
        else:
            if name not in definitions:
                raise ValueError(f"{module} defines no {name}, which a restart key names")
            nodes = definitions[name]
            lines = source.splitlines()
            chosen[(module, name)] = "\n".join(
                "\n".join(lines[min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", []))]) - 1:node.end_lineno])
                for node in nodes
            )
        for node in nodes:
            for item in ast.walk(node):
                if isinstance(item, ast.Name):
                    if item.id in definitions and item.id != name:
                        pending.append((module, item.id))
                    elif item.id in imports:
                        target, imported = imports[item.id]
                        pending.append((target, imported or "*"))
                elif isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id in imports:
                    target, imported = imports[item.value.id]
                    if imported is None:
                        pending.append((target, item.attr))
                elif isinstance(item, ast.ImportFrom) and item.level == 0 and item.module is not None and item.module.startswith("sv_pgs."):
                    pending.extend((item.module, alias.name) for alias in item.names)
    return hashlib.sha256(json.dumps({f"{module}:{name}": text for (module, name), text in sorted(chosen.items())}).encode()).hexdigest()


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class RestartStore:
    """One fit's restart units under ``directory`` (the module docstring): ``<kind>/<key>/`` with ``MANIFEST``. Names
    starting with "." are work in progress or released, which every reader and the mirror ignore."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def _record(path: Path) -> dict | None:
        """The unit at ``path``'s manifest where it is whole: the format, its own kind and key, and every sealed file
        at its recorded size (a relative path inside the unit); None otherwise."""
        try:
            record = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
            if record["format"] != FORMAT or record["kind"] != path.parent.name or record["key"] != path.name:
                return None
            if record["files"] is not None:
                for name, size in record["files"].items():
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts or name == MANIFEST or (path / relative).stat().st_size != size:
                        return None
            return record
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def units(self, kind: str) -> list[tuple[Path, dict]]:
        """Every whole unit of ``kind``, in key order."""
        parent = self.directory / kind
        if not parent.is_dir():
            return []
        found = [(path, self._record(path)) for path in sorted(parent.iterdir()) if path.is_dir() and not path.name.startswith(".")]
        return [(path, record) for path, record in found if record is not None]

    def unit(self, kind: str, key: str) -> tuple[Path, dict] | None:
        """The whole unit ``kind``/``key``, or None."""
        path = self.directory / kind / key
        record = self._record(path) if path.is_dir() else None
        return None if record is None else (path, record)

    def explain(self, kind: str, group: str, named: Mapping[str, str]) -> None:
        """Log why ``group``'s ``kind`` is made again: the parts that differ from its earlier unit's, or that there is none."""
        earlier = [record for _path, record in self.units(kind) if record["group"] == group]
        if not earlier:
            log(f"restart store: {kind} of {group}: no earlier unit; making it")
            return
        for record in earlier:
            changed = sorted(name for name in set(named) | set(record["facts"]) if named.get(name) != record["facts"].get(name))
            log(f"restart store: {kind} of {group} made again: {', '.join(changed) or 'its values'} changed since unit {record['key'][:16]}")

    def _install(self, staging: Path, record: dict) -> Path:
        """Write ``record`` into ``staging`` last and rename it into place; a whole unit already there stays."""
        manifest = staging / MANIFEST
        temporary = staging / f".{MANIFEST}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
        target = self.directory / record["kind"] / record["key"]
        if target.exists():
            if self._record(target) is not None:
                _remove(staging)
                return target
            _remove(target)
        os.rename(staging, target)
        return target

    @contextmanager
    def build(self, kind: str, key: str, *, group: str, named: Mapping[str, str], values: Mapping[str, Any] | None = None,
              supersede: bool = True) -> Iterator[Path]:
        """A hidden directory to make sealed unit ``kind``/``key`` in; on a clean exit its files are recorded with
        their sizes, the manifest written last and the directory renamed into place (``unit``), and, with
        ``supersede``, ``group``'s other units of the kind are released. A failure leaves no unit and removes its work."""
        parent = self.directory / kind
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key[:16]}.", dir=parent))
        try:
            yield staging
            files = {
                path.relative_to(staging).as_posix(): path.stat().st_size
                for path in sorted(staging.rglob("*")) if path.is_file()
            }
            self._install(staging, {
                "format": FORMAT, "kind": kind, "key": key, "group": group, "facts": dict(named), "values": dict(values or {}),
                "id": uuid.uuid4().hex, "files": files,
            })
        finally:
            if staging.exists():
                _remove(staging)
        if supersede:
            self._supersede(kind, key, group)

    def mutable(self, kind: str, key: str, *, group: str, named: Mapping[str, str], supersede: bool = False) -> Path:
        """Mutable unit ``kind``/``key``'s directory, made (empty, with its manifest) where it is not whole; with
        ``supersede``, ``group``'s other units of the kind are released, also where it was whole already (a stop, or a
        restore of a push cut short, can leave the older beside it)."""
        found = self.unit(kind, key)
        if found is not None:
            path = found[0]
        else:
            parent = self.directory / kind
            parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{key[:16]}.", dir=parent))
            path = self._install(staging, {
                "format": FORMAT, "kind": kind, "key": key, "group": group, "facts": dict(named), "values": {},
                "id": uuid.uuid4().hex, "files": None,
            })
        if supersede:
            self._supersede(kind, key, group)
        return path

    def _supersede(self, kind: str, key: str, group: str) -> None:
        for path, record in self.units(kind):
            if record["group"] == group and path.name != key:
                log(f"restart store: {kind} {path.name[:16]} of {group} superseded by {key[:16]}")
                _remove(path)

    def release(self, group: str | None = None) -> None:
        """Remove ``group``'s units of every kind (None: every group's but ``REQUEST``'s): its models are
        checkpointed, so nothing reads them again. The mirror removes their remote copies at its next push."""
        for kind in KINDS:
            for path, record in self.units(kind):
                if record["group"] == group or (group is None and record["group"] != REQUEST):
                    _remove(path)

    def clean(self) -> None:
        """Remove work in progress a stopped process left (hidden entries: an unfinished unit, a released one, an
        archive's temporary); only one fit writes a store at a time."""
        for parent in (self.directory / kind for kind in KINDS):
            if not parent.is_dir():
                continue
            for path in parent.iterdir():
                if path.name.startswith("."):
                    _remove(path)
                elif path.is_dir():
                    for inner in path.iterdir():
                        if inner.name.startswith("."):
                            _remove(inner)


def _remove(path: Path) -> None:
    """Hide ``path`` first (a rename: no reader sees it half removed), then remove it, after the cycles that can hold
    memory maps of its files are collected (``stage2_wiring._remove_group_scratch``)."""
    aside = path.with_name(f".released.{path.name.lstrip('.')}.{uuid.uuid4().hex}") if not path.name.startswith(".released.") else path
    os.rename(path, aside)
    gc.collect()
    if aside.is_dir():
        shutil.rmtree(aside)
    else:
        aside.unlink()


@dataclass(frozen=True)
class WarmStart:
    """Where ``full_data_fit.fit_full_data`` keeps each model's best mean-field start: ``store``'s ``eb`` units of
    ``group``, keyed by ``request`` (the fitting request without the engine's source: the store, the records' offsets,
    the problem and the prior's form) and the actual problem and prior of each solve (``state_key``)."""

    store: RestartStore
    group: str
    request: str


@dataclass(frozen=True)
class CodeCache:
    """Where ``store_block_source.StoreGenotypeBlockSource`` keeps each LD block's signed codes: ``store``'s ``codes``
    units of ``group``, keyed with ``store_key`` (the store's content digest, ``artifact.store_digest``)."""

    store: RestartStore
    group: str
    store_key: str


# ---------------------------------------------------------------------------
# Stage 0's statistics
# ---------------------------------------------------------------------------


def save_statistics(directory: Path, statistics: Any) -> None:
    """Stage 0's arrays beside its LD files (``directory``/ld): every field with its stored dtype and exact values, the
    tie map as its three arrays and the boundaries' cut costs per chromosome, never a pickle."""
    ties, boundaries = statistics.tie_map, statistics.boundaries
    arrays = {
        field.name: np.asarray(getattr(statistics, field.name)) for field in dataclasses.fields(statistics)
        if field.name not in ("ld", "tie_map", "boundaries")
    }
    arrays.update(tie_kept=ties.kept_indices, tie_reduced=ties.original_to_reduced, tie_signs=ties.signs)
    arrays.update({f"boundary_{field.name}": np.asarray(getattr(boundaries, field.name)) for field in dataclasses.fields(boundaries)
                   if field.name not in ("cut_costs", "chromosomes")})
    arrays["boundary_chromosomes"] = np.array(list(boundaries.chromosomes), dtype=np.str_)
    names = sorted(boundaries.cut_costs)
    arrays["cut_names"] = np.array(names, dtype=np.str_)
    arrays.update({f"cut{index}": np.asarray(boundaries.cut_costs[name]) for index, name in enumerate(names)})
    np.savez(directory / STATISTICS, **arrays)


def load_statistics(directory: Path) -> Any:
    """The statistics ``save_statistics`` wrote in ``directory``, its LD store opened from ``directory``/ld."""
    from sv_pgs.genotype_statistics import GenotypeSufficientStatistics, LdGramStore
    from sv_pgs.ld_partition import LdBlockBoundaries
    from sv_pgs.tie_map import _tie_map

    with np.load(directory / STATISTICS, allow_pickle=False) as saved:
        arrays = {name: np.array(saved[name]) for name in saved.files}
    ties = _tie_map(arrays.pop("tie_kept"), arrays.pop("tie_reduced"), arrays.pop("tie_signs"))
    names = [str(name) for name in arrays.pop("cut_names").tolist()]
    boundary = {
        "cut_costs": {name: arrays.pop(f"cut{index}") for index, name in enumerate(names)},
        "chromosomes": tuple(str(name) for name in arrays.pop("boundary_chromosomes").tolist()),
    }
    for field in dataclasses.fields(LdBlockBoundaries):
        if field.name not in boundary:
            boundary[field.name] = arrays.pop(f"boundary_{field.name}")
    boundary["block_cap"], boundary["forced_cuts"] = int(boundary["block_cap"]), int(boundary["forced_cuts"])
    arrays["sample_count"], arrays["covariate_rank"] = int(arrays["sample_count"]), int(arrays["covariate_rank"])
    return GenotypeSufficientStatistics(**arrays, tie_map=ties, boundaries=LdBlockBoundaries(**boundary), ld=LdGramStore(directory / "ld"))


# ---------------------------------------------------------------------------
# Empirical Bayes warm starts
# ---------------------------------------------------------------------------


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """A state or input schema as JSON with named arrays. Dataclasses are data, never executable pickle objects;
    derived fields (a binary site's metric key) are made again by their class, not loaded."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("a warm start's arrays cannot hold Python objects")
        name = f"array{len(arrays)}"
        arrays[name] = value
        return {"array": name}
    if dataclasses.is_dataclass(value):
        return {"class": type(value).__name__, "fields": {
            field.name: _encode(getattr(value, field.name), arrays) for field in dataclasses.fields(value) if field.init
        }}
    if isinstance(value, (tuple, list)):
        return [_encode(entry, arrays) for entry in value]
    if isinstance(value, dict):
        return {name: _encode(value[name], arrays) for name in sorted(value)}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"a warm start cannot hold a {type(value).__name__}")


def state_key(**parts: Any) -> str:
    """The digest of a warm start's actual inputs, arrays with their shapes and dtypes, and no source code or path: a
    start is solved and certified again by whatever engine reads it, so an engine edit keeps it."""
    arrays: dict[str, np.ndarray] = {}
    metadata = _encode(parts, arrays)
    return named_digest({"metadata": json.dumps(metadata, sort_keys=True), **arrays})


def save_warm_start(path: Path, key: str, hyperparameters: Any, state: Any) -> None:
    """One model's best refreshed q and its hyperparameters, checksummed, replacing the previous one atomically."""
    arrays: dict[str, np.ndarray] = {}
    metadata = {"format": WARM_FORMAT, "key": key, "hyperparameters": _encode(hyperparameters, arrays), "state": _encode(state, arrays)}
    metadata["digest"] = named_digest({"metadata": json.dumps(metadata, sort_keys=True), **arrays})
    save_archive(path, metadata, arrays)


def load_warm_start(path: Path, key: str, prior: Any, residual_length: int, sites: Any) -> tuple[Any, Any] | None:
    """A matching, whole start (hyperparameters, state), or None with the reason logged. The checksum covers the JSON
    and every array; shapes, signs and finite moments are checked before any oracle installs them, the residual's length
    is the route's (samples, or the band's groups), and a binary start's sites name the current labels and training
    rows. Its representation is checked, not its convergence: the first oracle call solves it again."""
    from sv_pgs.binary_likelihood import BernoulliSites
    from sv_pgs.full_data_fit import _MODEL_STATE_ARRAYS, _ModelState
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    classes = {cls.__name__: cls for cls in (BernoulliSites, _ModelState, MixtureHyperparameters)}
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format") != WARM_FORMAT or metadata.get("key") != key:
                raise ValueError("another fitting problem or state format")
            arrays = {name: np.array(archive[name]) for name in archive.files if name != "metadata"}
            digest = metadata.pop("digest")
            if digest != named_digest({"metadata": json.dumps(metadata, sort_keys=True), **arrays}):
                raise ValueError("its checksum differs")

        def decode(value: Any) -> Any:
            if not isinstance(value, dict):
                return value
            if "array" in value:
                return arrays[value["array"]]
            return classes[value["class"]](**{name: decode(entry) for name, entry in value["fields"].items()})

        hyperparameters, state = decode(metadata["hyperparameters"]), decode(metadata["state"])
        if not isinstance(hyperparameters, MixtureHyperparameters) or not isinstance(state, _ModelState):
            raise ValueError("it holds no mean-field start")
        for name in _MODEL_STATE_ARRAYS:
            value = getattr(state, name)
            size = residual_length if name == "residual" else prior.variant_count
            if value.dtype != np.dtype(np.float64) or value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"its {name} is not {size} finite float64 values")
        if np.any(state.variance < 0.0) or np.any(state.fourth < 0.0):
            raise ValueError("a variance or fourth moment is negative")
        scalars = (state.noise, state.effective, state.mean_move, state.noise_gain, state.elbo)
        if not np.all(np.isfinite(scalars)) or state.noise <= 0.0 or state.effective <= 0.0:
            raise ValueError("its scalars are not finite and positive")
        if (hyperparameters.coefficients.shape != (prior.coefficient_size,) or not np.all(np.isfinite(hyperparameters.coefficients))
                or hyperparameters.log_smoothing.shape != (len(prior.smoothing_blocks),) or np.any(np.isnan(hyperparameters.log_smoothing))):
            raise ValueError("its hyperparameters do not fit the prior")
        if (state.sites is None) != (sites is None):
            raise ValueError("another likelihood")
        if sites is not None and (
            not isinstance(state.sites, BernoulliSites) or state.sites.xi.shape != sites.xi.shape
            or not np.array_equal(state.sites.training, sites.training) or not np.array_equal(state.sites.labels, sites.labels) or state.noise != 1.0
        ):
            raise ValueError("another binary training problem")
        log(f"eb warm start: loaded {path.parent.name[:16]}/{path.name}; solving again at its saved hyperparameters")
        return hyperparameters, state
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, BadZipFile) as error:
        log(f"eb warm start: ignored {path.name} ({error})")
        return None

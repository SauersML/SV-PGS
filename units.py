"""The resume units on a run's disk: each EB solve's saved model states (d30ea8a's ``restart_store`` warm starts).

A unit is ``<run directory>/live_restart/<arm>/eb/<key>/``: ``UNIT.json`` (the fitting request it belongs to) and one
archive per model, ``model<i>.npz``. Each archive is JSON metadata and named non-object arrays with a checksum over
both, written aside and renamed into place after its bytes are durable, so a reader (or the mirror's push) sees the old
archive or the whole new one. Names starting with "." are work in progress, which nothing reads.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from sv_pgs.progress import log

ROOT = "live_restart"
"""The units' directory in the run directory, and their prefix in the run mirror."""
MANIFEST = "UNIT.json"
WARM_FORMAT = "svpgs-live-eb-state-v1"


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """A state or input schema as JSON with named arrays. Dataclasses are data, never executable pickle objects;
    derived fields (a binary site's metric key) are made again by their class, not loaded."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("a saved state's arrays cannot hold Python objects")
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
    raise TypeError(f"a saved state cannot hold a {type(value).__name__}")


def _digest(metadata: Any, arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over the JSON and each named array's dtype, shape and bytes, read in place (no copy of a large array)."""
    hasher = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        hasher.update(f"{name}{array.dtype.str}{array.shape}".encode())
        hasher.update(array.reshape(-1).view(np.uint8))
    return hasher.hexdigest()


def state_key(**parts: Any) -> str:
    """The digest of a saved state's actual inputs, arrays with their shapes and dtypes, and no source code or path."""
    arrays: dict[str, np.ndarray] = {}
    return _digest(_encode(parts, arrays), arrays)


def _save_archive(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """One NPZ of JSON and arrays, renamed into place only after its bytes are durable; a failed write leaves the old
    archive and no temporary (``fit_checkpoint.save_checkpoint``'s writer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, metadata=np.array(json.dumps(metadata)), **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_archive(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A saved state's metadata (without its checksum) and arrays, where its format is this module's and its checksum
    holds; raises otherwise."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if not isinstance(metadata, dict) or metadata.get("format") != WARM_FORMAT:
            raise ValueError("another state format")
        arrays = {name: np.array(archive[name]) for name in archive.files if name != "metadata"}
    digest = metadata.pop("digest", None)
    if digest != _digest(metadata, arrays):
        raise ValueError("its checksum differs")
    return metadata, arrays


def whole(path: Path, name: str) -> bool:
    """Whether the unit file ``name`` fetched from the mirror to ``path`` is whole: a saved state whose checksum holds,
    or a unit's manifest naming its request."""
    try:
        if name.endswith(".npz"):
            _read_archive(path)
            return True
        return name == MANIFEST and isinstance(json.loads(path.read_text(encoding="utf-8")).get("request"), str)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, BadZipFile):
        return False


def save_state(path: Path, key: str, hyperparameters: Any, state: Any) -> None:
    """One model's best refreshed q and its hyperparameters, checksummed, replacing the previous one atomically."""
    arrays: dict[str, np.ndarray] = {}
    metadata = {"format": WARM_FORMAT, "key": key, "hyperparameters": _encode(hyperparameters, arrays), "state": _encode(state, arrays)}
    metadata["digest"] = _digest(metadata, arrays)
    _save_archive(path, metadata, arrays)


def load_state(path: Path, key: str, prior: Any, residual_length: int, sites: Any) -> tuple[Any, Any] | None:
    """A matching, whole saved state (hyperparameters, state), or None with the reason logged. The checksum covers the
    JSON and every array; shapes, signs and finite moments are checked before any oracle installs them, the residual's
    length is the route's (samples, or the band's groups), and a binary state's sites name the current labels and
    training rows. Its representation is checked, not its convergence: the first oracle call solves it again."""
    from sv_pgs.binary_likelihood import BernoulliSites
    from sv_pgs.full_data_fit import _MODEL_STATE_ARRAYS, _ModelState
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    classes = {cls.__name__: cls for cls in (BernoulliSites, _ModelState, MixtureHyperparameters)}
    try:
        metadata, arrays = _read_archive(path)
        if metadata.get("key") != key:
            raise ValueError("another fitting problem")

        def decode(value: Any) -> Any:
            if not isinstance(value, dict):
                return value
            if "array" in value:
                return arrays[value["array"]]
            return classes[value["class"]](**{name: decode(entry) for name, entry in value["fields"].items()})

        hyperparameters, state = decode(metadata["hyperparameters"]), decode(metadata["state"])
        if not isinstance(hyperparameters, MixtureHyperparameters) or not isinstance(state, _ModelState):
            raise TypeError("it holds no mean-field state")
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
        return hyperparameters, state
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, BadZipFile) as error:
        log(f"live resume: ignored {path.parent.name[:16]}/{path.name} ({error})")
        return None


def unit_directory(root: Path, arm: str, key: str, request: str) -> Path:
    """Unit ``key``'s directory in ``root``/``arm``/eb, made with its manifest (the request it belongs to) where it is
    not there yet."""
    path = root / arm / "eb" / key
    if not (path / MANIFEST).is_file():
        path.mkdir(parents=True, exist_ok=True)
        (path / f".{MANIFEST}").write_text(json.dumps({"request": request}), encoding="utf-8")
        os.replace(path / f".{MANIFEST}", path / MANIFEST)
    return path


def release(root: Path, arm: str, request: str) -> int:
    """Remove ``arm``'s units of ``request`` (its models are checkpointed, so nothing reads them again); the mirror's next
    push removes their remote copies. Returns how many were removed."""
    parent = root / arm / "eb"
    removed = 0
    if not parent.is_dir():
        return removed
    for path in sorted(parent.iterdir()):
        if path.name.startswith(".") or not path.is_dir():
            continue
        try:
            owner = json.loads((path / MANIFEST).read_text(encoding="utf-8")).get("request")
        except (OSError, ValueError, AttributeError):
            continue
        if owner == request:
            # Hidden first (a rename: no reader or push sees it half removed), then removed.
            aside = path.with_name(f".released.{path.name}.{uuid.uuid4().hex}")
            os.rename(path, aside)
            gc.collect()
            shutil.rmtree(aside, ignore_errors=True)
            removed += 1
    return removed

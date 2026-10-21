"""Atomic, non-pickle checkpoints for completed folds, including their original draw laws.

Only complete fits are reused here; a mean-field start (``restart_store.save_warm_start``) shares the atomic archive
writer and is solved and certified again, never reused as a fit. The caller binds each checkpoint to a digest of the
engine's code, the store and the fitting problem, and keeps it in the restart store (``restart_store``).
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from sv_pgs.draw_laws import DenseDraws, ProductMixtureDraws, law_from_arrays
from sv_pgs.config import TraitType

FORMAT = "svpgs-fold-checkpoint-v1"


def save_checkpoint(path: Path, fit: Any, key: str, elapsed_seconds: float) -> None:
    arrays: dict[str, np.ndarray] = {}

    def encode(value: Any) -> Any:
        if isinstance(value, TraitType):
            return {"trait_type": value.value}
        if isinstance(value, np.ndarray):
            name = f"array{len(arrays)}"
            if value.dtype.hasobject:
                raise ValueError("checkpoint arrays cannot contain Python objects")
            arrays[name] = value
            return {"array": name}
        if isinstance(value, (DenseDraws, ProductMixtureDraws)):
            return {"law": value.kind, "arrays": {name: encode(array) for name, array in value.arrays().items()}}
        if dataclasses.is_dataclass(value):
            return {"class": type(value).__name__, "fields": {field.name: encode(getattr(value, field.name)) for field in dataclasses.fields(value)}}
        if isinstance(value, tuple):
            return {"tuple": [encode(entry) for entry in value]}
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")

    metadata = {"format": FORMAT, "key": key, "elapsed_seconds": elapsed_seconds, "fit": encode(fit)}
    save_archive(path, metadata, arrays)


def save_archive(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """One NPZ of JSON and non-object arrays, renamed into place only after its bytes are durable: a reader (or the run
    mirror) sees the old archive or the whole new one, and a failed write leaves the old one and no temporary."""
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("checkpoint arrays cannot contain Python objects")
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


def load_checkpoint(path: Path, key: str) -> Any:
    from sv_pgs.fast_scoring import ScoringModel
    from sv_pgs.full_data_fit import FitCertificate
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters
    from sv_pgs.stage2_wiring import _ModelFit

    classes = {cls.__name__: cls for cls in (_ModelFit, ScoringModel, FitCertificate, MixtureHyperparameters)}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("format") != FORMAT or metadata.get("key") != key:
            raise ValueError(f"checkpoint {path} belongs to a different fitting problem")

        def decode(value: Any) -> Any:
            if not isinstance(value, dict):
                return value
            if "array" in value:
                return np.array(archive[value["array"]])
            if "trait_type" in value:
                return TraitType(value["trait_type"])
            if "tuple" in value:
                return tuple(decode(entry) for entry in value["tuple"])
            if "law" in value:
                return law_from_arrays(value["law"], {name: decode(array) for name, array in value["arrays"].items()})
            if value.get("class") not in classes:
                raise ValueError("unknown checkpoint class")
            return classes[value["class"]](**{name: decode(entry) for name, entry in value["fields"].items()})

        result = decode(metadata["fit"])
        if not isinstance(result, _ModelFit):
            raise ValueError("checkpoint does not hold a completed model")
        return result


def checkpoint_timings(directory: Path) -> list[dict[str, Any]]:
    """Small timing records for cost projections, without loading coefficients or participant data."""
    records = []
    for path in sorted(directory.glob("checkpoints/*/model*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        records.append({"model_index": int(path.stem[5:]), "elapsed_seconds": metadata["elapsed_seconds"], "key": metadata["key"]})
    return records

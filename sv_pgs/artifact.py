"""A fitted SV-PGS model on disk: every (trait, fold) scoring model, the fit's certificate, its prior hyperparameters and
its provenance. A model loads and scores without the fit that made it.

Layout of a model directory::

    model.json   format, model names and trait types, covariate names, provenance, and the name of every array
    arrays.npz   every array, by name

``save_model`` writes a sibling temporary directory, flushes both files, and renames the directory into place, so
a reader finds either no model or a complete one. An existing model is never overwritten. ``load_model`` refuses
any model whose format, names, shapes or values are not exactly what ``save_model`` writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import MANIFEST_FILE, DosageStore, read_manifest
from sv_pgs.fast_scoring import (
    GeneticScores,
    ScoringModel,
    ScoringPlan,
    posterior_predictive_probability,
    score_genetic,
    score_linear_predictor,
)
from sv_pgs.scale_mixture_ep import MixtureHyperparameters

MODEL_FORMAT = "svpgs-model v1"
_METADATA = "model.json"
_ARRAYS = "arrays.npz"
_SCORING_FIELDS = ("store_rows", "signed_means", "signed_scales", "coefficients", "posterior_draws", "alpha")
CERTIFICATE_STATUS = "outer_criterion_met"
"""The certificate term every fit records (``full_data_fit.FitCertificate``): per model, whether the outer loop's own
stopping criterion was met. It is not certification (M04), and a model whose certificate lacks it says nothing at all
about how its fit ended, so it is not a model this package wrote."""


@dataclass(frozen=True)
class Provenance:
    """What produced a model: the exact code, the exact store, its variant layout and the exact training cohort.

    ``code_digest`` is the SHA-256 of the ``sv_pgs`` sources, ``store_digest`` that of the training store's
    ``MANIFEST.json``, ``sites_digest`` that of its chromosomes, record counts and site digests (the layout the
    scoring models' store rows index; any store scored with the model must have the same one),
    ``cohort_digest`` that of the training research IDs, and ``offset_digest`` that of the records' log reliabilities
    the prior was given (``offset_digest(None)`` when it read the store's own ``quality`` column). Digests only, no
    identifiers.
    """

    code_digest: str
    store_digest: str
    sites_digest: str
    cohort_digest: str
    offset_digest: str


def code_digest() -> str:
    """SHA-256 over every ``sv_pgs`` source file, in path order, each prefixed by its path."""
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def store_digest(store_root: str | Path) -> str:
    """SHA-256 of the store's manifest file."""
    return hashlib.sha256((Path(store_root) / MANIFEST_FILE).read_bytes()).hexdigest()


def sites_digest(store_root: str | Path) -> str:
    """SHA-256 of the store's variant layout: its chromosomes in order, their record counts and site digests."""
    manifest = read_manifest(Path(store_root))
    layout = {
        "chromosomes": manifest["chromosomes"],
        "record_counts": manifest["record_counts"],
        "sites_md5": [manifest["sites_md5"][chromosome] for chromosome in manifest["chromosomes"]],
    }
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()


def cohort_digest(research_ids: Sequence[str]) -> str:
    """SHA-256 over the training research IDs, sorted and newline-joined."""
    values = sorted(str(value) for value in research_ids)
    if len(set(values)) != len(values):
        raise ValueError("research_ids repeats a participant.")
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def offset_digest(log_variance_offset: np.ndarray | None) -> str:
    """SHA-256 of the records' little-endian float64 log reliabilities; of no bytes when the store's own were used."""
    values = b"" if log_variance_offset is None else np.asarray(log_variance_offset, dtype="<f8").tobytes()
    return hashlib.sha256(values).hexdigest()


@dataclass(frozen=True)
class FittedModel:
    """One fitted model per (trait, fold) training set, in ``model_names`` order.

    ``covariate_names`` are the union of every model's covariates, and ``covariate_columns`` [models, covariates] the
    ones each model adjusted for (with the intercept, always); a model's alpha is exactly 0 on the others.
    ``certificate`` holds every per-model term of the fit's certificate as an array with one leading entry per model,
    ``fit_counts`` its whole-fit counts, and ``refusals`` the reasons the fit refused trial steps, across all models;
    ``noise_variance`` is each quantitative model's residual variance, used in its predictive variance.
    """

    model_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    covariate_columns: BoolArray
    scoring: tuple[ScoringModel, ...]
    noise_variance: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    certificate: Mapping[str, F64Array]
    fit_counts: Mapping[str, int]
    refusals: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        model_count = len(self.model_names)
        if model_count == 0 or len(set(self.model_names)) != model_count:
            raise ValueError("a model needs distinct model names.")
        if len(self.scoring) != model_count or len(self.hyperparameters) != model_count:
            raise ValueError("scoring and hyperparameters need one entry per model.")
        noise = np.asarray(self.noise_variance)
        if noise.shape != (model_count,) or noise.dtype != np.float64 or not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
            raise ValueError("noise_variance must be positive float64 with one entry per model.")
        columns = np.asarray(self.covariate_columns)
        if columns.shape != (model_count, len(self.covariate_names)) or columns.dtype != np.bool_:
            raise ValueError("covariate_columns must be bool [models, covariates].")
        for model, adjusted in zip(self.scoring, columns):
            if model.alpha.shape[0] != len(self.covariate_names) + 1:
                raise ValueError("each model's alpha must hold the intercept and one entry per covariate.")
            if np.any(model.alpha[1:][~adjusted] != 0.0):
                raise ValueError("a model's alpha must be 0 on the covariates it did not adjust for.")
        if CERTIFICATE_STATUS not in self.certificate:
            raise ValueError(f"the certificate must carry {CERTIFICATE_STATUS!r}; a model that says nothing about its fit is not one.")
        for name, values in self.certificate.items():
            term = np.asarray(values)
            if term.shape[:1] != (model_count,):
                raise ValueError(f"certificate term {name!r} needs one entry per model.")
            # A padded term carries NaN past each model's count, so only the kind is checked, never finiteness.
            if term.dtype.kind not in "fiub":
                raise ValueError(f"certificate term {name!r} must be a numeric or boolean array.")

    @property
    def trait_types(self) -> tuple[TraitType, ...]:
        return tuple(model.trait_type for model in self.scoring)


def save_model(path: str | Path, model: FittedModel) -> None:
    """Write ``model`` to the new directory ``path``, atomically."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target} exists; a model is never overwritten.")
    arrays: dict[str, np.ndarray] = {"noise_variance": model.noise_variance, "covariate_columns": np.asarray(model.covariate_columns)}
    for index, scoring in enumerate(model.scoring):
        for field_name in _SCORING_FIELDS:
            arrays[f"scoring/{index}/{field_name}"] = getattr(scoring, field_name)
    for index, hyperparameters in enumerate(model.hyperparameters):
        arrays[f"hyperparameters/{index}/coefficients"] = hyperparameters.coefficients
        arrays[f"hyperparameters/{index}/log_smoothing"] = hyperparameters.log_smoothing
    for name, values in model.certificate.items():
        arrays[f"certificate/{name}"] = np.asarray(values)
    metadata = {
        "format": MODEL_FORMAT,
        "model_names": list(model.model_names),
        "trait_types": [trait_type.value for trait_type in model.trait_types],
        "covariate_names": list(model.covariate_names),
        "predictive_intercept_shifts": [scoring.predictive_intercept_shift for scoring in model.scoring],
        "certificate_terms": sorted(model.certificate),
        "fit_counts": {name: int(count) for name, count in model.fit_counts.items()},
        "refusals": list(model.refusals),
        "provenance": {
            "code_digest": model.provenance.code_digest,
            "store_digest": model.provenance.store_digest,
            "sites_digest": model.provenance.sites_digest,
            "cohort_digest": model.provenance.cohort_digest,
            "offset_digest": model.provenance.offset_digest,
        },
        "arrays": sorted(arrays),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    with open(staging / _ARRAYS, "wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    with open(staging / _METADATA, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(staging, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _required(metadata: Mapping[str, Any], key: str, kind: type) -> Any:
    if key not in metadata or not isinstance(metadata[key], kind):
        raise ValueError(f"model.json has no valid {key!r}.")
    return metadata[key]


def load_model(path: str | Path) -> FittedModel:
    """Read a model written by ``save_model``; anything else raises ``ValueError``."""
    directory = Path(path)
    with open(directory / _METADATA, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or metadata.get("format") != MODEL_FORMAT:
        raise ValueError(f"{directory} is not a {MODEL_FORMAT!r} model.")
    model_names = tuple(_required(metadata, "model_names", list))
    trait_types = tuple(TraitType(value) for value in _required(metadata, "trait_types", list))
    covariate_names = tuple(_required(metadata, "covariate_names", list))
    shifts = _required(metadata, "predictive_intercept_shifts", list)
    terms = _required(metadata, "certificate_terms", list)
    fit_counts = {str(name): int(count) for name, count in _required(metadata, "fit_counts", dict).items()}
    refusals = tuple(str(reason) for reason in _required(metadata, "refusals", list))
    provenance = _required(metadata, "provenance", dict)
    names = _required(metadata, "arrays", list)
    if not len(model_names) == len(trait_types) == len(shifts):
        raise ValueError("model.json lists models, trait types and intercept shifts of different lengths.")
    with np.load(directory / _ARRAYS, allow_pickle=False) as archive:
        if sorted(archive.files) != sorted(names):
            raise ValueError("arrays.npz does not hold exactly the arrays model.json lists.")
        arrays = {name: np.array(archive[name]) for name in archive.files}
    scoring = tuple(
        ScoringModel(
            **{field_name: arrays[f"scoring/{index}/{field_name}"] for field_name in _SCORING_FIELDS},
            trait_type=trait_type,
            predictive_intercept_shift=float(shift),
        )
        for index, (trait_type, shift) in enumerate(zip(trait_types, shifts))
    )
    hyperparameters = tuple(
        MixtureHyperparameters(
            coefficients=arrays[f"hyperparameters/{index}/coefficients"],
            log_smoothing=arrays[f"hyperparameters/{index}/log_smoothing"],
        )
        for index in range(len(model_names))
    )
    return FittedModel(
        model_names=model_names,
        covariate_names=covariate_names,
        covariate_columns=arrays["covariate_columns"],
        scoring=scoring,
        noise_variance=arrays["noise_variance"],
        hyperparameters=hyperparameters,
        certificate={name: arrays[f"certificate/{name}"] for name in terms},
        fit_counts=fit_counts,
        refusals=refusals,
        provenance=Provenance(
            code_digest=str(provenance["code_digest"]),
            store_digest=str(provenance["store_digest"]),
            sites_digest=str(provenance["sites_digest"]),
            cohort_digest=str(provenance["cohort_digest"]),
            offset_digest=str(provenance["offset_digest"]),
        ),
    )


class StoreCodeBlocks:
    """A ``DosageStore`` as ``fast_scoring.CodeBlockSource``: its own read-ahead ring, all samples."""

    def __init__(self, store: DosageStore, budget: ComputeBudget) -> None:
        self.store = store
        self.budget = budget

    @property
    def sample_count(self) -> int:
        return int(self.store.n_samples)

    def iter_code_blocks(self, variant_ranges, buffers):
        yield from self.store.iter_codes(variant_ranges, None, self.budget)


@dataclass(frozen=True)
class Prediction:
    """Predictions [samples, models] for the scored samples.

    ``genetic`` holds the posterior-mean genetic scores and their K-draw variances under each model's drawing law
    (``fast_scoring``: the posterior's on the full-data route, the fitted product approximation's on the mean-field
    route, so a spread reported here is the posterior's only in the first case);
    ``linear_predictor`` adds the intercept and covariate effects. ``predictive_mean`` is the linear predictor for a
    quantitative trait and P(y = 1) for a binary one; ``predictive_variance`` is the genetic variance plus the noise
    variance for a quantitative trait and p(1 - p) for a binary one.
    """

    genetic: GeneticScores
    linear_predictor: F64Array
    predictive_mean: F64Array
    predictive_variance: F64Array


def predict(model: FittedModel, store: DosageStore, sample_indices: np.ndarray, covariates: F64Array, budget: ComputeBudget) -> Prediction:
    """Score ``model`` on the store samples ``sample_indices``, whose covariates are ``covariates`` in the model's
    ``covariate_names`` order without the intercept, from one read of the store."""
    if sites_digest(store.root) != model.provenance.sites_digest:
        raise ValueError(f"{store.root} has a different variant layout than the store the model was fitted on.")
    samples = np.asarray(sample_indices, dtype=np.int64)
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    if covariate_matrix.shape != (samples.shape[0], len(model.covariate_names)):
        raise ValueError("covariates must be [samples, the model's covariates].")
    genetic = score_genetic(StoreCodeBlocks(store, budget), ScoringPlan.from_models(model.scoring), budget, samples)
    linear_predictor = score_linear_predictor(genetic.means, covariate_matrix, model.scoring)
    predictive_mean = np.empty_like(linear_predictor)
    predictive_variance = np.empty_like(linear_predictor)
    for index, scoring in enumerate(model.scoring):
        if scoring.trait_type == TraitType.BINARY:
            probability = posterior_predictive_probability(
                linear_predictor[:, index], genetic.variances[:, index], scoring.predictive_intercept_shift
            )
            predictive_mean[:, index] = probability
            predictive_variance[:, index] = probability * (1.0 - probability)
        else:
            predictive_mean[:, index] = linear_predictor[:, index]
            genetic_variance = genetic.variances[:, index]
            if not np.all(np.isfinite(genetic_variance)):
                raise ValueError(f"model {model.model_names[index]!r} has no posterior draws, so no predictive variance.")
            predictive_variance[:, index] = genetic_variance + model.noise_variance[index]
    return Prediction(genetic=genetic, linear_predictor=linear_predictor, predictive_mean=predictive_mean, predictive_variance=predictive_variance)


def write_predictions(model_path: str | Path, store_path: str | Path, people_path: str | Path, output_path: str | Path, budget: ComputeBudget) -> None:
    """Score the people in ``people_path`` (an NPZ holding ``sample_indices`` [n], their store columns, and
    ``covariates`` [n, k] in the model's covariate order without the intercept) and write an NPZ of
    ``model_names`` and the prediction's arrays [n, models]. An existing output is never overwritten."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"{output} exists; predictions are never overwritten.")
    model = load_model(model_path)
    with np.load(people_path, allow_pickle=False) as people:
        if sorted(people.files) != ["covariates", "sample_indices"]:
            raise ValueError("the people file must hold exactly sample_indices and covariates.")
        sample_indices = np.array(people["sample_indices"])
        covariates = np.array(people["covariates"])
    prediction = predict(model, DosageStore.open(store_path), sample_indices, covariates, budget)
    with open(output, "xb") as handle:
        np.savez(
            handle,
            model_names=np.array(model.model_names),
            genetic_mean=prediction.genetic.means,
            genetic_variance=prediction.genetic.variances,
            draw_counts=np.array(prediction.genetic.draw_counts, dtype=np.int64),
            linear_predictor=prediction.linear_predictor,
            predictive_mean=prediction.predictive_mean,
            predictive_variance=prediction.predictive_variance,
        )
        handle.flush()
        os.fsync(handle.fileno())

"""The public entry point: a dosage store and a cohort in, a fitted model out.

Every column of ``targets`` and ``training`` is one model: a trait on one training set (a fold's training rows, or all
of a trait's observed rows). ``full_data_fit.fit_models`` runs Stage 0 once over the union of the training rows and
fits every model by EP-EB from its learned prior; this module checks the cohort, adds the provenance, and splits the
certificate into per-model terms, whole-fit counts and refusals for ``artifact.FittedModel``.

``FitRequest`` is the whole input of one fit, checked and normalized by its own constructor. ``fit`` takes nothing
else, so a caller that builds a request (``workspace_pipeline``'s fit step) either builds this exact object or fails
where it builds it: there is no keyword list to drift from the fitter's signature.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.artifact import (
    FittedModel,
    Provenance,
    code_digest,
    cohort_digest,
    offset_digest,
    problem_digest,
    save_model,
    sites_digest,
    store_digest,
)
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.full_data_fit import FitCertificate
# RUN-ONLY (run/svpgs-bench-1): e2e's full_data_fit.fit_models replaces this wiring.
from sv_pgs.stage2_wiring import fit_models

# MODEL.md section 4: the scorer's K posterior draws, which also set every certificate tolerance (1/(2K) nats).
DRAW_COUNT = 64


@dataclasses.dataclass(frozen=True)
class CertificateParts:
    terms: dict[str, np.ndarray]
    counts: dict[str, int]
    refusals: tuple[str, ...]


def certificate_parts(certificate: FitCertificate, model_count: int) -> CertificateParts:
    """The certificate as per-model arrays, whole-fit counts and refusal reasons.

    A per-model sequence of varying length (one entry per interior weight) is padded with NaN to the longest and
    stored with ``<name>_count``, the number of valid entries of each model, so a pad is never read as a value.

    A term the fit did not record is None, which is refused here by name: an artifact says what the fit established,
    so a missing term is never written as a number (a scalar-like None reached ``int(None)`` before) and never left
    out in silence.
    """
    terms: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    refusals: tuple[str, ...] = ()
    for field in dataclasses.fields(certificate):
        value = getattr(certificate, field.name)
        if value is None:
            raise ValueError(f"certificate term {field.name!r} was not recorded by this fit; an artifact never invents one.")
        if isinstance(value, tuple) and all(isinstance(entry, str) for entry in value):
            refusals = value
        elif isinstance(value, tuple):
            lengths = np.array([np.asarray(entry).shape[0] for entry in value], dtype=np.int64)
            if lengths.shape != (model_count,):
                raise ValueError(f"certificate term {field.name!r} needs one sequence per model.")
            padded = np.full((model_count, int(lengths.max(initial=0))), np.nan)
            for model, entry in enumerate(value):
                padded[model, : lengths[model]] = entry
            terms[field.name] = padded
            terms[f"{field.name}_count"] = lengths
        elif np.ndim(value) == 0:
            counts[field.name] = int(value)
        else:
            terms[field.name] = np.asarray(value)
    return CertificateParts(terms=terms, counts=counts, refusals=refusals)


@dataclasses.dataclass(frozen=True)
class FitRequest:
    """One fit's whole input, checked and normalized here, so ``fit`` never validates a second time.

    Cohort row i is the store sample ``store_columns[i]`` with id ``research_ids[i]``; ``covariates`` [n, k] holds the
    named covariates of every model without the intercept, as ``artifact.predict`` takes them, and
    ``covariate_columns`` [models, k] the ones each model adjusts for besides the intercept; ``targets`` and
    ``training`` are [n, models], and a model's non-training targets are never read. ``log_variance_offset`` [store
    records] is each record's log measurement reliability, log r^2 <= 0, the prior's variance offset (-inf: the record
    carries no signal); None takes r^2 from the store's ``quality`` column. Stage 0 keeps its LD blocks under
    ``work_dir``.

    The fit has no measurement-object and no target-variance term: the measurement model enters as the records'
    ``log_variance_offset`` (and as the store's own recalibrated dosages), and every target is taken as measured
    exactly. A caller with a term this request has no field for cannot pass it, so nothing is dropped in silence.
    """

    store: DosageStore
    store_columns: I64Array
    covariates: F64Array
    covariate_names: tuple[str, ...]
    covariate_columns: BoolArray
    targets: F64Array
    training: BoolArray
    model_names: tuple[str, ...]
    trait_types: tuple[TraitType, ...]
    research_ids: tuple[str, ...]
    log_variance_offset: F64Array | None
    budget: ComputeBudget
    work_dir: Path
    seed: int

    def __post_init__(self) -> None:
        """Check every field's rank, kind, values and agreement, and only then convert.

        Nothing is converted before it is checked, so an input never changes meaning on the way in: a float array of
        store columns or a seed is refused rather than truncated, a mask of codes is refused rather than read as
        "train on every nonzero row", and a wrong rank is named rather than raising where a shape is indexed.
        """
        set_field = object.__setattr__
        set_field(self, "covariate_names", tuple(str(name) for name in self.covariate_names))
        set_field(self, "model_names", tuple(str(name) for name in self.model_names))
        set_field(self, "trait_types", tuple(self.trait_types))
        set_field(self, "research_ids", tuple(str(research_id) for research_id in self.research_ids))
        set_field(self, "work_dir", Path(self.work_dir))
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer.")
        set_field(self, "seed", int(self.seed))
        model_count = len(self.model_names)
        if model_count == 0 or len(set(self.model_names)) != model_count:
            raise ValueError("a fit needs at least one model, with distinct model names.")
        columns = np.asarray(self.store_columns)
        if columns.ndim != 1 or columns.dtype.kind not in "iu":
            raise ValueError("store_columns must be a 1-D integer array of distinct store samples.")
        columns = columns.astype(np.int64)
        row_count = columns.shape[0]
        if np.unique(columns).shape[0] != row_count or np.any(columns < 0) or np.any(columns >= self.store.n_samples):
            raise ValueError("store_columns must name distinct store samples.")
        if len(self.research_ids) != row_count or len(set(self.research_ids)) != row_count:
            raise ValueError("research_ids needs one distinct id per cohort row.")
        covariate_matrix = np.asarray(self.covariates)
        if covariate_matrix.ndim != 2 or covariate_matrix.dtype.kind not in "fiu":
            raise ValueError("covariates must be finite [cohort rows, named covariates].")
        covariate_matrix = covariate_matrix.astype(np.float64)
        if covariate_matrix.shape != (row_count, len(self.covariate_names)) or not np.all(np.isfinite(covariate_matrix)):
            raise ValueError("covariates must be finite [cohort rows, named covariates].")
        adjusted = np.asarray(self.covariate_columns)
        if adjusted.shape != (model_count, len(self.covariate_names)) or adjusted.dtype != np.bool_:
            raise ValueError("covariate_columns must be bool [models, named covariates].")
        training_mask = np.asarray(self.training)
        if training_mask.dtype != np.bool_:
            raise ValueError("training must be a bool mask; a count or a code is not a training flag.")
        target_matrix = np.asarray(self.targets)
        if target_matrix.ndim != 2 or target_matrix.dtype.kind not in "fiu":
            raise ValueError("targets and training must be [cohort rows, models].")
        target_matrix = target_matrix.astype(np.float64)
        if target_matrix.shape != (row_count, model_count) or training_mask.shape != target_matrix.shape:
            raise ValueError("targets and training must be [cohort rows, models].")
        if len(self.trait_types) != model_count:
            raise ValueError("trait_types needs one entry per model.")
        # Before Stage 0 reads the store: a model with no training row has nothing to fit and nothing to predict.
        untrained = [name for name, count in zip(self.model_names, training_mask.sum(axis=0)) if count == 0]
        if untrained:
            raise ValueError(f"models {untrained} have no training rows.")
        if not np.all(np.isfinite(target_matrix[training_mask])):
            raise ValueError("every training target must be finite.")
        if self.log_variance_offset is not None:
            offset = np.asarray(self.log_variance_offset)
            if offset.ndim != 1 or offset.dtype.kind not in "fiu":
                raise ValueError("log_variance_offset must be a log reliability <= 0 for every store record.")
            offset = offset.astype(np.float64)
            if offset.shape != (self.store.n_variants,) or np.any(np.isnan(offset)) or np.any(offset > 0.0):
                raise ValueError("log_variance_offset must be a log reliability <= 0 for every store record.")
            set_field(self, "log_variance_offset", offset)
        for model, trait_type in enumerate(self.trait_types):
            if trait_type == TraitType.BINARY and not np.all(np.isin(target_matrix[training_mask[:, model], model], (0.0, 1.0))):
                raise ValueError(f"binary model {self.model_names[model]!r} has training targets other than 0 and 1.")
        set_field(self, "store_columns", columns)
        set_field(self, "covariates", covariate_matrix)
        set_field(self, "targets", target_matrix)
        set_field(self, "training", training_mask)
        set_field(self, "covariate_columns", adjusted)

    @property
    def model_count(self) -> int:
        return len(self.model_names)


def fit(request: FitRequest) -> FittedModel:
    """Fit every model of ``request`` on its training rows and return the artifact."""
    row_count, model_count = request.store_columns.shape[0], request.model_count
    fitted = fit_models(
        store=request.store,
        store_columns=request.store_columns,
        covariates=np.column_stack([np.ones(row_count), request.covariates]),
        covariate_columns=np.column_stack([np.ones(model_count, dtype=bool), request.covariate_columns]),
        targets=np.where(request.training, request.targets, 0.0),
        training=request.training,
        trait_types=request.trait_types,
        log_variance_offset=request.log_variance_offset,
        budget=request.budget,
        work_dir=request.work_dir,
        seed=request.seed,
        draw_count=DRAW_COUNT,
    )
    parts = certificate_parts(fitted.certificate, model_count)
    trained = request.training.any(axis=1)
    return FittedModel(
        model_names=request.model_names,
        covariate_names=request.covariate_names,
        covariate_columns=request.covariate_columns,
        scoring=tuple(fitted.scoring),
        noise_variance=np.asarray(fitted.noise_variance, dtype=np.float64),
        hyperparameters=tuple(fitted.hyperparameters),
        certificate=parts.terms,
        fit_counts=parts.counts,
        refusals=parts.refusals,
        provenance=Provenance(
            code_digest=code_digest(),
            store_digest=store_digest(request.store.root),
            sites_digest=sites_digest(request.store.root),
            cohort_digest=cohort_digest([research_id for research_id, kept in zip(request.research_ids, trained) if kept]),
            offset_digest=offset_digest(request.log_variance_offset),
            problem_digest=problem_digest(
                research_ids=request.research_ids,
                store_columns=request.store_columns,
                model_names=request.model_names,
                trait_types=request.trait_types,
                covariate_names=request.covariate_names,
                covariate_columns=request.covariate_columns,
                covariates=request.covariates,
                targets=request.targets,
                training=request.training,
                draw_count=DRAW_COUNT,
                seed=request.seed,
            ),
            # One digest over the models' own prior schemas, in model order.
            prior_digest=hashlib.sha256("\n".join(fitted.prior_digests).encode()).hexdigest(),
        ),
    )


def cohort_seed(store_root: str | Path, research_ids: Sequence[str]) -> int:
    """The fit's seed from the store and the cohort, so the same inputs always give the same model."""
    return int.from_bytes(hashlib.sha256(f"{store_digest(store_root)}\n{cohort_digest(research_ids)}".encode()).digest(), "big")


def write_model(store_path: str | Path, cohort_path: str | Path, model_path: str | Path, budget: ComputeBudget) -> None:
    """``sv-pgs fit``: fit the cohort file's models on the store and save them to the new directory ``model_path``.

    The cohort NPZ holds research_ids [n], store_columns [n], covariates [n, k] without the intercept,
    covariate_names [k], covariate_columns [m, k], targets [n, m], training [n, m], model_names [m] and
    trait_types [m]; nothing else. Each record's reliability is the store's ``quality`` column.
    """
    target = Path(model_path)
    if target.exists():
        raise FileExistsError(f"{target} exists; a model is never overwritten.")
    names = (
        "research_ids",
        "store_columns",
        "covariates",
        "covariate_names",
        "covariate_columns",
        "targets",
        "training",
        "model_names",
        "trait_types",
    )
    with np.load(cohort_path, allow_pickle=False) as cohort:
        if sorted(cohort.files) != sorted(names):
            raise ValueError(f"{cohort_path} must hold exactly {sorted(names)}.")
        arrays = {name: np.array(cohort[name]) for name in names}
    research_ids = [str(research_id) for research_id in arrays["research_ids"]]
    store = DosageStore.open(store_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.work.", dir=target.parent))
    try:
        model = fit(
            FitRequest(
                store=store,
                store_columns=arrays["store_columns"],
                covariates=arrays["covariates"],
                covariate_names=tuple(str(name) for name in arrays["covariate_names"]),
                covariate_columns=arrays["covariate_columns"],
                targets=arrays["targets"],
                training=arrays["training"],
                model_names=tuple(str(name) for name in arrays["model_names"]),
                trait_types=tuple(TraitType(str(value)) for value in arrays["trait_types"]),
                research_ids=research_ids,
                log_variance_offset=None,
                budget=budget,
                work_dir=work_dir,
                seed=cohort_seed(store_path, research_ids),
            )
        )
    finally:
        shutil.rmtree(work_dir)
    save_model(target, model)

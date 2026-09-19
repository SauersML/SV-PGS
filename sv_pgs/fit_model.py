"""The public entry point: a dosage store and a cohort in, a fitted model out.

Every column of ``targets`` and ``training`` is one model: a trait on one training set (a fold's training rows, or all
of a trait's observed rows). ``full_data_fit.fit_models`` runs Stage 0 once over the union of the training rows and
fits every model by EP-EB from its learned prior; this module checks the cohort, adds the provenance, and splits the
certificate into per-model terms, whole-fit counts and refusals for ``artifact.FittedModel``.
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
    save_model,
    sites_digest,
    store_digest,
)
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.full_data_fit import FitCertificate, fit_models

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
    """
    terms: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    refusals: tuple[str, ...] = ()
    for field in dataclasses.fields(certificate):
        value = getattr(certificate, field.name)
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


def fit(
    *,
    store: DosageStore,
    store_columns: I64Array,
    covariates: F64Array,
    covariate_names: Sequence[str],
    targets: F64Array,
    training: BoolArray,
    model_names: Sequence[str],
    trait_types: Sequence[TraitType],
    research_ids: Sequence[str],
    log_variance_offset: F64Array | None,
    budget: ComputeBudget,
    work_dir: str | Path,
    seed: int,
) -> FittedModel:
    """Fit every model on its training rows and return the artifact.

    Cohort row i is the store sample ``store_columns[i]`` with id ``research_ids[i]``; ``covariates`` [n, k] holds the
    named covariates without the intercept, as ``artifact.predict`` takes them; ``targets`` and ``training`` are
    [n, models], and a model's non-training targets are never read. ``log_variance_offset`` [store records] is each
    record's log measurement reliability, log r^2 <= 0, the prior's variance offset (-inf: the record carries no
    signal); None takes r^2 from the store's ``quality`` column. Stage 0 keeps its LD blocks under ``work_dir``.
    """
    columns = np.asarray(store_columns, dtype=np.int64)
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    target_matrix = np.asarray(targets, dtype=np.float64)
    training_mask = np.asarray(training, dtype=bool)
    row_count, model_count = columns.shape[0], len(model_names)
    if columns.ndim != 1 or np.unique(columns).shape[0] != row_count or np.any(columns < 0) or np.any(columns >= store.n_samples):
        raise ValueError("store_columns must name distinct store samples.")
    if len(research_ids) != row_count or len(set(research_ids)) != row_count:
        raise ValueError("research_ids needs one distinct id per cohort row.")
    if covariate_matrix.shape != (row_count, len(covariate_names)) or not np.all(np.isfinite(covariate_matrix)):
        raise ValueError("covariates must be finite [cohort rows, named covariates].")
    if target_matrix.shape != (row_count, model_count) or training_mask.shape != target_matrix.shape:
        raise ValueError("targets and training must be [cohort rows, models].")
    if len(trait_types) != model_count:
        raise ValueError("trait_types needs one entry per model.")
    if not np.all(np.isfinite(target_matrix[training_mask])):
        raise ValueError("every training target must be finite.")
    if log_variance_offset is not None:
        offset = np.asarray(log_variance_offset, dtype=np.float64)
        if offset.shape != (store.n_variants,) or np.any(np.isnan(offset)) or np.any(offset > 0.0):
            raise ValueError("log_variance_offset must be a log reliability <= 0 for every store record.")
    for model, trait_type in enumerate(trait_types):
        if trait_type == TraitType.BINARY and not np.all(np.isin(target_matrix[training_mask[:, model], model], (0.0, 1.0))):
            raise ValueError(f"binary model {model_names[model]!r} has training targets other than 0 and 1.")
    fitted = fit_models(
        store=store,
        store_columns=columns,
        covariates=np.column_stack([np.ones(row_count), covariate_matrix]),
        targets=np.where(training_mask, target_matrix, 0.0),
        training=training_mask,
        trait_types=tuple(trait_types),
        log_variance_offset=None if log_variance_offset is None else np.asarray(log_variance_offset, dtype=np.float64),
        budget=budget,
        work_dir=Path(work_dir),
        seed=seed,
        draw_count=DRAW_COUNT,
    )
    parts = certificate_parts(fitted.certificate, model_count)
    trained = training_mask.any(axis=1)
    return FittedModel(
        model_names=tuple(model_names),
        covariate_names=tuple(covariate_names),
        scoring=tuple(fitted.scoring),
        noise_variance=np.asarray(fitted.noise_variance, dtype=np.float64),
        hyperparameters=tuple(fitted.hyperparameters),
        certificate=parts.terms,
        fit_counts=parts.counts,
        refusals=parts.refusals,
        provenance=Provenance(
            code_digest=code_digest(),
            store_digest=store_digest(store.root),
            sites_digest=sites_digest(store.root),
            cohort_digest=cohort_digest([research_id for research_id, kept in zip(research_ids, trained) if kept]),
            offset_digest=offset_digest(log_variance_offset),
        ),
    )


def cohort_seed(store_root: str | Path, research_ids: Sequence[str]) -> int:
    """The fit's seed from the store and the cohort, so the same inputs always give the same model."""
    return int.from_bytes(hashlib.sha256(f"{store_digest(store_root)}\n{cohort_digest(research_ids)}".encode()).digest(), "big")


def write_model(store_path: str | Path, cohort_path: str | Path, model_path: str | Path, budget: ComputeBudget) -> None:
    """``sv-pgs fit``: fit the cohort file's models on the store and save them to the new directory ``model_path``.

    The cohort NPZ holds research_ids [n], store_columns [n], covariates [n, k] without the intercept,
    covariate_names [k], targets [n, m], training [n, m], model_names [m] and trait_types [m]; nothing else. Each
    record's reliability is the store's ``quality`` column.
    """
    target = Path(model_path)
    if target.exists():
        raise FileExistsError(f"{target} exists; a model is never overwritten.")
    names = ("research_ids", "store_columns", "covariates", "covariate_names", "targets", "training", "model_names", "trait_types")
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
            store=store,
            store_columns=arrays["store_columns"],
            covariates=arrays["covariates"],
            covariate_names=[str(name) for name in arrays["covariate_names"]],
            targets=arrays["targets"],
            training=arrays["training"],
            model_names=[str(name) for name in arrays["model_names"]],
            trait_types=[TraitType(str(value)) for value in arrays["trait_types"]],
            research_ids=research_ids,
            log_variance_offset=None,
            budget=budget,
            work_dir=work_dir,
            seed=cohort_seed(store_path, research_ids),
        )
    finally:
        shutil.rmtree(work_dir)
    save_model(target, model)

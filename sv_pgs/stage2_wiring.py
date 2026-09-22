"""RUN-ONLY (run/svpgs-bench-1, fit-api): store -> fitted models, until e2e's ``full_data_fit.fit_models`` replaces it.

This is e2e's tests/test_full_data_fit wiring behind the agreed ``fit_models`` signature, run once per model:
Stage 0 on the model's own training rows and covariate columns, the start lattice from its single-variant
likelihoods, the prior with one class per variant class present, the records' log reliabilities as offsets and the store's
other sidecar columns as its annotation groups (``annotation_design``), the dual Gaussian, ``fit_full_data`` by the mean-field fixed
points (``full_data_fit._FullDataMeanField``) and ``scoring_models``. Models are fitted separately, so there is no cross-trait pooling of the prior's hyperparameters.
Quantitative traits only: ``fit_full_data`` has no binary likelihood yet.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.artifact import named_digest
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.full_data_fit import FitCertificate, block_grams, covariate_residual_variance, fit_full_data, scoring_models, stage0_lattice
from sv_pgs.genotype_buffers import build_sample_layout
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import BLOCK_CAP_STEP, DosageStoreTileSource, compute_genotype_statistics, stage0_block_cap
from sv_pgs.imputation_reliability import checked_log_reliability
from sv_pgs.annotation_design import annotation_design
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import MixtureHyperparameters, scale_mixture_prior
from sv_pgs.store_block_source import StoreGenotypeBlockSource

_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class FittedModels:
    scoring: list[ScoringModel]
    noise_variance: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    certificate: FitCertificate
    # Each model's prior schema (``_prior_schema_digest``), which its fitted hyperparameters alone do not identify.
    prior_digests: tuple[str, ...]


def _seed(seed: int, *keys: int) -> int:
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1, dtype=np.uint64)[0])


RELIABILITY_COLUMNS = ("quality", "r2_truth")
"""Sidecar columns that are a measurement's reliability, the prior's offset, never one of its annotations."""


def store_log_reliability(store: DosageStore) -> F64Array:
    """Each record's log r^2 from the store's ``quality`` column, its reported imputation r^2.

    A store without the column carries no measurement, and the only model that lets its records be
    fitted at all is perfect measurement, r^2 = 1. A hard-called benchmark store means exactly that;
    a store that has lost its column does not, and the two are told apart by reading the line this
    logs, not by the prior. A quality the column does hold must be a reliability
    (``checked_log_reliability``): a value above 1, below 0 or missing is corrupt metadata.
    """
    quality = store.variant_table.annotations.get("quality")
    if quality is None:
        log(f"stage2 wiring: no quality column, so all {store.n_variants:,} records are taken as measured exactly (r^2 = 1)")
        return np.zeros(store.n_variants)
    with np.errstate(divide="ignore", invalid="ignore"):
        offsets = np.log(np.asarray(quality, dtype=np.float64))
    return checked_log_reliability(offsets, "the store's quality column")


def stage0_candidates(store: DosageStore, training_columns: I64Array, log_reliability: F64Array, config: ModelConfig) -> I64Array:
    """The store rows Stage 0 reads: every record with signal, less those Stage 0 would find inactive.

    When the training samples are all of the store's, the sidecar's code sums are their sums, so a record's training
    frequency and variance are known before the pass; one that is monomorphic, or rarer than the minimum minor
    allele frequency by more than one code unit's frequency (a margin that keeps every record Stage 0's own float
    test could keep), is dropped here rather than after its Grams are formed. Otherwise every record is a candidate.
    """
    rows = np.flatnonzero(np.isfinite(log_reliability)).astype(np.int64)
    count = training_columns.shape[0]
    if count != store.n_samples:
        return rows
    table = store.variant_table
    sums = np.asarray(table.sum_code, dtype=np.float64)[rows]
    variance_numerator = count * np.asarray(table.sum_code2, dtype=np.float64)[rows] - sums * sums
    frequency = sums / (2.0 * SIGNED_CODE_OFFSET * count)
    margin = 1.0 / (2.0 * SIGNED_CODE_OFFSET * count)
    keep = (variance_numerator > 0.0) & (np.minimum(frequency, 1.0 - frequency) >= config.minimum_minor_allele_frequency - margin)
    return rows[keep]


def _block_cap(store: DosageStore, candidates: I64Array, training_columns: I64Array, covariate_count: int, budget: ComputeBudget) -> int:
    """The memory's largest cap, but no larger than a chromosome's candidate count (in cap steps): Stage 0's buffers
    grow with the cap, and a block never holds more than its chromosome's candidates.

    There is at least one candidate: ``_fit_one`` returns the null genetic model before Stage 0 when there is none."""
    sample_groups = np.full(store.n_samples, -1, dtype=np.int64)
    sample_groups[training_columns] = 0
    memory_cap = stage0_block_cap(budget, build_sample_layout(sample_groups), covariate_count + 1)
    chromosome_of_row = np.searchsorted(np.asarray(store.chromosome_starts), candidates, side="right") - 1
    largest = int(np.bincount(chromosome_of_row).max())
    return min(memory_cap, -(-largest // BLOCK_CAP_STEP) * BLOCK_CAP_STEP)


def _merged_certificate(certificates: Sequence[FitCertificate]) -> FitCertificate:
    """One certificate over models fitted one at a time: per-model terms stacked, whole-fit counts summed, refusals
    joined with their model's index."""
    values = {}
    for field in dataclasses.fields(FitCertificate):
        parts = [getattr(certificate, field.name) for certificate in certificates]
        if field.name == "refusals":
            values[field.name] = tuple(f"model {model}: {reason}" for model, part in enumerate(parts) for reason in part)
        elif isinstance(parts[0], tuple):
            values[field.name] = tuple(entry for part in parts for entry in part)
        elif np.ndim(parts[0]) == 0:
            values[field.name] = sum(int(part) for part in parts)
        else:
            values[field.name] = np.concatenate([np.asarray(part) for part in parts])
    return FitCertificate(**values)


@dataclass(frozen=True)
class _ModelFit:
    """One model's fit: what ``fit_models`` collects per model."""

    scoring: ScoringModel
    noise: float
    hyperparameters: MixtureHyperparameters
    certificate: FitCertificate
    prior_digest: str


def _prior_schema_digest(
    *, nodes: F64Array, floor: F64Array, top: F64Array, class_index: I64Array, offsets: F64Array, rows: I64Array,
    annotations: F64Array | None = None, annotation_names: Sequence[str] = (),
) -> str:
    """The prior's schema, which its fitted coefficients do not carry: the lattice its density is written on, the
    variant class of each row the prior covers, those rows in the store, the offset each one was given, and the
    annotation design theta multiplies (its columns by name). Without it the saved hyperparameters name no density
    (``artifact.Provenance.prior_digest``). Empty arrays for a model with no prior at all (the null genetic model)."""
    return named_digest({
        "nodes": nodes, "floor": floor, "top": top, "class_index": class_index, "offsets": offsets, "rows": rows,
        "annotations": np.zeros((0, 0)) if annotations is None else annotations,
        "annotation_names": np.array(list(annotation_names), dtype=str),
    })


@dataclass(frozen=True)
class _CovariateFit:
    """The covariates' own least squares on one model's training rows.

    ``degrees`` is n - rank(C), by the rank of the covariates themselves rather than their column count, and
    ``noise`` the residual sum of squares over it. ``explained`` is whether the residual is zero to the covariates'
    own numerical rank tolerance, max(n, k) eps ||y||, the rule Stage 0 uses for the covariate span
    (``small_n._covariate_basis``): the targets then lie in the covariates' span to working precision, so the data
    say nothing about a residual for a genetic model to explain.
    """

    alpha: F64Array
    noise: float
    degrees: int
    explained: bool


def _covariate_least_squares(covariates: F64Array, targets: F64Array) -> _CovariateFit:
    alpha, _sums, rank, _singular = np.linalg.lstsq(covariates, targets, rcond=None)
    residual = targets - covariates @ alpha
    residual_sum = float(residual @ residual)
    degrees = int(targets.shape[0] - rank)
    resolution = max(covariates.shape) * _EPSILON * float(np.linalg.norm(targets))
    return _CovariateFit(
        alpha=alpha,
        noise=residual_sum / degrees if degrees > 0 else np.inf,
        degrees=degrees,
        explained=residual_sum <= resolution * resolution,
    )


def _null_genetic_model(covariate_fit: _CovariateFit, draw_count: int, reason: str) -> _ModelFit:
    """The model of a training set with no genetic column to fit: the covariates alone, and no variant.

    This is the fit's answer, not a failure. A training set can legitimately have no genetic column: every record's
    reliability can be zero, every candidate can be monomorphic on its own rows, or the covariates can already
    explain its targets to working precision. The predictor is then the covariate part alone, the prior is empty,
    and there is nothing approximated and nothing to search, so every certificate term is 0 and ``reason`` says
    which case it was. The noise variance is the covariates' own residual variance as computed, never a floor and
    never rounded up: where the covariates span the targets it can be 0, and the predictive variance is 0 with it.
    """
    log(f"stage2 wiring: null genetic model ({reason}); the covariates alone, noise {covariate_fit.noise:.6g}")
    scoring = ScoringModel(
        store_rows=np.zeros(0, dtype=np.int64),
        signed_means=np.zeros(0),
        signed_scales=np.zeros(0),
        coefficients=np.zeros(0),
        posterior_draws=np.zeros((0, draw_count)),
        alpha=covariate_fit.alpha,
        trait_type=TraitType.QUANTITATIVE,
        predictive_intercept_shift=0.0,
    )
    certificate = FitCertificate(
        remaining_gain=np.zeros(1),
        newton_decrement=np.zeros(1),
        smoothing_gradient=np.zeros(1),
        stationarity_steps=(np.zeros(0),),
        stationarity_errors=(np.zeros(0),),
        mean_move=np.zeros(1),
        draw_tolerance=np.full(1, 1.0 / draw_count),
        noise_gain=np.zeros(1),
        mean_error=np.zeros(1),
        information_bound=np.zeros(1),
        information_tolerance=np.zeros(1),
        undecided_blocks=0,
        negative_sites=np.zeros(1, dtype=np.int64),
        # p_eff = 0: there are no effects, so the prediction tolerance p_eff / K is 0 and the move is 0 with it.
        effective_effects=np.zeros(1),
        outer_iterations=np.zeros(1, dtype=np.int64),
        halvings=np.zeros(1, dtype=np.int64),
        prediction_move=np.zeros(1),
        prediction_tolerance=np.zeros(1),
        unresolved=np.zeros(1, dtype=np.int64),
        refusals=(reason,),
        outer_history=((),),
        refreshes=0,
        passes=0,
        # There is no outer loop to stop and no fixed point to perturb: the covariate least squares is exact.
        outer_criterion_met=np.ones(1, dtype=bool),
    )
    return _ModelFit(
        scoring=scoring,
        noise=covariate_fit.noise,
        hyperparameters=MixtureHyperparameters(coefficients=np.zeros(0), log_smoothing=np.zeros(0)),
        certificate=certificate,
        prior_digest=_prior_schema_digest(
            nodes=np.zeros(0), floor=np.zeros(0), top=np.zeros(0), class_index=np.zeros(0, np.int64),
            offsets=np.zeros(0), rows=np.zeros(0, np.int64),
        ),
    )


def _fit_one(
    store: DosageStore,
    training_columns: I64Array,
    covariates: F64Array,
    targets: F64Array,
    log_reliability: F64Array,
    budget: ComputeBudget,
    work_dir: Path,
    seed: int,
    draw_count: int,
) -> _ModelFit:
    """One quantitative model on the sorted store columns ``training_columns``, whose covariates (intercept first) and
    targets follow them; the null genetic model where the training set has no genetic column to fit."""
    # SPEC 1fca1cf: no variant is filtered by rarity or any threshold (only a derived bound may leave one out).
    config = ModelConfig(minimum_minor_allele_frequency=0.0)
    # Before the store is read: what the covariates alone leave, which decides whether there is anything to fit.
    covariate_fit = _covariate_least_squares(covariates, targets)
    if covariate_fit.degrees <= 0:
        raise ValueError(
            f"{targets.shape[0]} training rows against covariates of rank {targets.shape[0] - covariate_fit.degrees} leave no "
            "residual degrees of freedom, so the noise variance is not identified."
        )
    if covariate_fit.explained:
        return _null_genetic_model(covariate_fit, draw_count, "the covariates explain every training target to working precision")
    candidates = stage0_candidates(store, training_columns, log_reliability, config)
    if candidates.shape[0] == 0:
        return _null_genetic_model(covariate_fit, draw_count, "no store record carries signal on these training rows")
    block_cap = _block_cap(store, candidates, training_columns, covariates.shape[1], budget)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, candidates), training_columns, covariates, targets[:, None], config, budget, block_cap, work_dir / "ld"
    )
    if np.asarray(statistics.active_rows).shape[0] == 0:
        return _null_genetic_model(covariate_fit, draw_count, "every candidate record is monomorphic on these training rows")
    kept_rows = np.asarray(statistics.active_rows, dtype=np.int64)[np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)]
    # The prior is over every active row: tie members keep their own class and offset (tie_members; review-mathbugs T1).
    member_rows = np.asarray(statistics.active_rows, dtype=np.int64)
    offsets = log_reliability[member_rows]
    _classes, class_index = np.unique(store.variant_table.variant_class[member_rows], return_inverse=True)
    table = store.variant_table
    annotations = annotation_design(
        {name: np.asarray(values)[member_rows] for name, values in table.annotations.items()},
        table.annotation_legends,
        class_index=class_index.astype(np.int64),
        exclude=RELIABILITY_COLUMNS,
    )
    log(f"stage2 wiring: {annotations.design.shape[1]} annotation columns in {len(annotations.groups)} groups: {', '.join(annotations.names) or 'none'}")
    mask = np.zeros((store.n_samples, 1))
    mask[training_columns, 0] = 1.0
    store_targets = np.zeros((store.n_samples, 1))
    store_targets[training_columns, 0] = targets
    store_covariates = np.zeros((store.n_samples, covariates.shape[1]))
    store_covariates[training_columns] = covariates
    start_noise = float(covariate_residual_variance(store_targets, mask, store_covariates)[0])
    tolerance = 0.5 / draw_count
    nodes, floor, top = stage0_lattice(statistics, 0, start_noise, offsets, tolerance)
    prior = scale_mixture_prior(
        class_index=class_index.astype(np.int64),
        log_variance_offset=offsets,
        annotation_design=annotations.design,
        annotation_groups=annotations.groups,
        nodes=nodes,
        floor=floor,
        top=top,
    )
    # The block source's read workspace and the fit's dense per-block jobs are live together: each gets half.
    share = budget.working_bytes // 2
    source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, budget, share))
    gaussian = DualGaussian(
        source=source,
        training=mask,
        targets=store_targets,
        offsets=np.zeros((store.n_samples, 1)),
        covariates=store_covariates,
        grams=block_grams(statistics, start_noise),
        probe_count=draw_count,
        seed=_seed(seed, 0),
    )
    # The mean-field fixed points: EP's refused nearly every call on the wiring store ("the EP refreshes' updates line
    # up with no contraction ... the full-data route has no double loop", 59 of 65 calls, 2026-09-21).
    fit = fit_full_data(
        gaussian=gaussian, statistics=statistics, prior=prior, draw_count=draw_count, working_bytes=share, seed=_seed(seed, 1), inference="mean_field"
    )
    (scoring,) = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], draw_count, seed=_seed(seed, 2))
    log(f"stage2 wiring: {kept_rows.shape[0]:,} reduced columns in {statistics.ld.block_count} blocks (cap {block_cap}), {training_columns.shape[0]:,} training samples")
    return _ModelFit(
        scoring=scoring,
        noise=float(np.asarray(fit.noise_variance)[0]),
        hyperparameters=fit.hyperparameters[0],
        certificate=fit.certificate,
        prior_digest=_prior_schema_digest(
            nodes=nodes, floor=np.array([floor]), top=np.array([top]), class_index=class_index.astype(np.int64),
            offsets=offsets, rows=member_rows, annotations=annotations.design, annotation_names=annotations.names,
        ),
    )


def fit_models(
    *,
    store: DosageStore,
    store_columns: I64Array,
    covariates: F64Array,
    covariate_columns: BoolArray,
    targets: F64Array,
    training: BoolArray,
    trait_types: Sequence[TraitType],
    log_variance_offset: F64Array | None,
    budget: ComputeBudget,
    work_dir: Path,
    seed: int,
    draw_count: int,
) -> FittedModels:
    """Every model of ``fit_model.fit``, one at a time (see the module docstring)."""
    if any(trait_type != TraitType.QUANTITATIVE for trait_type in trait_types):
        raise NotImplementedError("run/svpgs-bench-1 fits quantitative traits only: fit_full_data has no binary likelihood yet.")
    log_reliability = (
        store_log_reliability(store)
        if log_variance_offset is None
        else checked_log_reliability(log_variance_offset, "the caller's log_variance_offset")
    )
    scoring, noise, hyperparameters, certificates, prior_digests = [], [], [], [], []
    for model in range(training.shape[1]):
        rows = np.flatnonzero(training[:, model])
        order = np.argsort(store_columns[rows], kind="stable")
        rows = rows[order]
        adjusted = np.asarray(covariate_columns[model], dtype=bool)
        model_dir = Path(work_dir) / f"model{model}"
        model_dir.mkdir()
        fit = _fit_one(
            store,
            np.asarray(store_columns[rows], dtype=np.int64),
            np.asarray(covariates[rows][:, adjusted], dtype=np.float64),
            np.asarray(targets[rows, model], dtype=np.float64),
            log_reliability,
            budget,
            model_dir,
            _seed(seed, model),
            draw_count,
        )
        alpha = np.zeros(adjusted.shape[0])
        alpha[adjusted] = fit.scoring.alpha
        scoring.append(dataclasses.replace(fit.scoring, alpha=alpha))
        noise.append(fit.noise)
        hyperparameters.append(fit.hyperparameters)
        certificates.append(fit.certificate)
        prior_digests.append(fit.prior_digest)
    return FittedModels(
        scoring=scoring,
        noise_variance=np.array(noise),
        hyperparameters=tuple(hyperparameters),
        certificate=_merged_certificate(certificates),
        prior_digests=tuple(prior_digests),
    )

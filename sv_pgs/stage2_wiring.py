"""Store -> fitted models: the production wiring of Stage 0, the prior and Stage 2, run once per group of models.

The models that train on the same rows with the same covariates and the same likelihood are one group
(``model_groups``): Stage 0's Grams, its LD blocks and band, the candidate records and the prior's schema depend on the
rows, the covariates and the records, never on a target, so the group's models share them and the dual solver's passes
(each further model adds right-hand-side columns, not passes). Each group gets:
Stage 0 on its training rows and covariate columns with every model's target as a column, the start lattice covering its
models' single-variant likelihoods, the prior with one class per variant class present, the records' unit offsets and
frequency function from the store's unit contract (``store_measurement``), the store's other sidecar columns as its
annotation groups (``annotation_design``), the dual Gaussian with a column per model, ``fit_full_data`` by the
mean-field fixed points (``full_data_fit._FullDataMeanField``) and ``scoring_models``. Stage 0's pass keeps the training
codes of every record it reads (``training_codes``): Stage 2's resident codes are filled from them, every device at once,
before the band's costs are measured, and the far field's columns are read from them, so no step after Stage 0 reads
the store while they are held (a restarted fit's Stage 0, far field and codes are its restart units', which read none).
Each model keeps its own hyperparameters, noise, lattice check, certificate and prior digest, as when fitted alone: a
group shares the store's reads, never a hyperparameter, so there is no cross-trait pooling of the prior. A group's
models share its probes' and draws' seed and start lattice. The finite numerical tolerances and local empirical-Bayes maxima can also affect
agreement with separate fits. A band-accelerated fit continues on the sample route before its final acceptance check.
A binary model (0/1 targets) is a group of its own, fitted on its Polya-Gamma bound (``binary_likelihood``), on the same
route; its reported noise variance is its whitened problem's known unit noise, which its predictive never reads.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.artifact import named_digest
from sv_pgs.fit_checkpoint import load_checkpoint, save_checkpoint
from sv_pgs.binary_likelihood import POLYA_GAMMA_MEAN_AT_ZERO, BernoulliSites, calibrated_shift, fit_binary_covariates
from sv_pgs.logistic_ep import assert_no_separation
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.device_plan import ShardedDualSource
from sv_pgs.dosage_store import VARIANT_CLASSES, DosageStore
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.eb_monitor import MONITOR_SWITCH, EbMonitor, EvaluationCodes, EvaluationRows, GroupEvaluation, monitor_on
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.gram_space import BandStore, GramBand, build_band_store, far_field_scale, pass_costs
from sv_pgs.full_data_fit import FitCertificate, FullDataFit, block_grams, covariate_residual_variance, fit_full_data, scoring_models, stage0_lattice, state_digest
from sv_pgs.marginal_variances import certificate_level
from sv_pgs.memory_broker import memory_scope
from sv_pgs.model_acceptance import acceptance
from sv_pgs.genotype_buffers import build_sample_layout
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import (
    BLOCK_CAP_STEP, DosageStoreTileSource, GenotypeSufficientStatistics, LdGramStore, ProjectedLdBlock, compute_genotype_statistics,
    largest_block_cap, stage0_block_cap,
)
from sv_pgs.imputation_reliability import ColumnMeasurement, checked_log_reliability
from sv_pgs.annotation_design import (
    BY_CLASS_ANNOTATIONS, LOG_SCALE_ANNOTATIONS, annotation_design, frequency_annotation, ld_score_annotation, smooth_shift_record,
)
from sv_pgs.progress import log
from sv_pgs.restart_store import (
    REQUEST, CodeCache, RestartStore, WarmStart, definition_sources, facts, load_statistics, module_sources, save_statistics, unit_key,
)
from sv_pgs.scale_mixture_ep import Cavity, MixtureHyperparameters, ScaleMixturePrior, extended_lattice, halved_lattice, lattice_check, scale_mixture_prior, transfer_hyperparameters
from sv_pgs.store_block_source import StoreGenotypeBlockSource
from sv_pgs.training_codes import CodeTee, TrainingCodes, code_cache_on

_EPSILON = float(np.finfo(np.float64).eps)


def _remove_group_scratch(directory: Path) -> None:
    """Release unreachable solver/ledger cycles before deleting their memory-mapped LD files.

    NFS keeps an unlinked but mapped file as .nfs*, so rmtree otherwise fails with ENOTEMPTY after a successful fit.
    Refcounting alone cannot release closures and cache-eviction callbacks that refer back to their owners.
    """
    gc.collect()
    shutil.rmtree(directory)


def _move_lattice(
    prior: ScaleMixturePrior, hyperparameters: Sequence[MixtureHyperparameters],
    move: Callable[[ScaleMixturePrior, MixtureHyperparameters], tuple[ScaleMixturePrior, MixtureHyperparameters]],
) -> tuple[ScaleMixturePrior, list[MixtureHyperparameters]]:
    """Move a group's shared prior once, transferring each model's own coefficients to it."""
    moved, first = move(prior, hyperparameters[0])
    return moved, [first, *(transfer_hyperparameters(prior, moved, item) for item in hyperparameters[1:])]


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


_FAR_FIELD_KEY = len(("dual solver", "fit", "scoring"))
"""The far-field scale's draws' seed key: after the dual solver's (0), the fit's (1) and the scoring's (2)."""


RELIABILITY_COLUMNS = ("quality", "r2_truth")
"""Sidecar columns that are a measurement's reliability, read by the unit contract (``store_measurement``), never annotations."""

IDENTIFIER_COLUMNS = ("tr_locus",)
"""Sidecar columns that name a record's locus (the workspace pipeline's TR locus id): a label, never a quantity, so
never a smooth's argument."""


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


def store_measurement(store: DosageStore, log_variance_offset: F64Array | None) -> ColumnMeasurement:
    """The unit contract of every store record (``imputation_reliability.ColumnMeasurement``), from the store's own
    metadata: its raw unit (one ALT allele, or one copy for a ``VariantClass.COPY_NUMBER`` record, whose value is
    CN - modal CN), its encoding scale ``codes_per_unit``, calibration slope 1 (the store holds calibrated conditional
    means: a GLIMPSE2 or Beagle DS, the measurement step's rewritten D*, or a hard call), and the log reliability from
    the caller's ``log_variance_offset`` (the measurement model's) or else the store's ``quality`` column, named as its
    source."""
    table = store.variant_table
    copy_number = np.asarray(table.variant_class) == VARIANT_CLASSES.index(VariantClass.COPY_NUMBER)
    if log_variance_offset is None:
        reliability, source = store_log_reliability(store), (
            "the store's quality column (reported imputation r^2)" if "quality" in table.annotations else "none: every record measured exactly"
        )
    else:
        reliability, source = checked_log_reliability(log_variance_offset, "the caller's log_variance_offset"), "the caller's log reliability (the measurement model's)"
    # Each record's label is a reference to one of two shared strings. A string per record (np.where's fixed-width array,
    # then a new object for each of its entries, ~72 bytes a record) was ~14 GB built and ~9 GB held through the fit at
    # the All of Us genome's 126.7M records, for the same labels; np.full makes a string per entry too, fill does not.
    raw_unit = np.empty(store.n_variants, dtype=object)
    raw_unit.fill("ALT allele")
    raw_unit[copy_number] = "copy"
    return ColumnMeasurement.build(
        store.n_variants, log_reliability=reliability, codes_per_unit=table.codes_per_unit, raw_unit=raw_unit, reliability_source=source,
    )


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


LD_STORE_BYTES_PER_VARIANT_AND_CAP = 2 * np.dtype(np.float32).itemsize
"""Stage 0's LD store per candidate and unit of block cap: each variant's float32 row of its block's Gram and of its
coupling to the block before, each at most the cap wide."""


def _cap_limits(store: DosageStore, candidates: I64Array, training_columns: I64Array, covariate_count: int, budget: ComputeBudget) -> tuple[int, int]:
    """The largest cap the memory allows, and the largest chromosome's candidate count in cap steps (``_block_cap``)."""
    sample_groups = np.full(store.n_samples, -1, dtype=np.int64)
    sample_groups[training_columns] = 0
    memory_cap = stage0_block_cap(budget, build_sample_layout(sample_groups), covariate_count + 1)
    chromosome_of_row = np.searchsorted(np.asarray(store.chromosome_starts), candidates, side="right") - 1
    largest = int(np.bincount(chromosome_of_row).max())
    return memory_cap, -(-largest // BLOCK_CAP_STEP) * BLOCK_CAP_STEP


def _block_cap(
    store: DosageStore, candidates: I64Array, training_columns: I64Array, covariate_count: int, budget: ComputeBudget, work_dir: Path,
) -> int:
    """The largest cap the memory and the disk allow, but no larger than a chromosome's candidate count (in cap steps):
    Stage 0's buffers grow with the cap, a block never holds more than its chromosome's candidates, and the LD store
    Stage 0 writes under ``work_dir`` is about candidates x cap x LD_STORE_BYTES_PER_VARIANT_AND_CAP bytes. The summary
    band is cut from that store and written beside it, never larger than it, so the store may take half the disk's
    free space. The memory alone set the cap before, which at 20M candidates on an A100 VM asked 3.6 TB of disk per
    model and failed when the disk filled, hours into Stage 0.

    There is at least one candidate: ``_fit_group`` returns the null genetic models before Stage 0 when there is none."""
    memory_cap, chromosome_cap = _cap_limits(store, candidates, training_columns, covariate_count, budget)
    free_bytes = shutil.disk_usage(work_dir).free
    try:
        disk_cap = largest_block_cap(free_bytes // 2, lambda cap: int(candidates.shape[0]) * cap * LD_STORE_BYTES_PER_VARIANT_AND_CAP)
    except MemoryError as error:
        raise OSError(f"{work_dir}'s {free_bytes / 1e9:.1f} GB free cannot hold Stage 0's LD store and band for {candidates.shape[0]:,} candidates: {error}") from error
    cap = min(memory_cap, disk_cap, chromosome_cap)
    log(f"stage2 wiring: block cap {cap:,} (memory {memory_cap:,}, disk {disk_cap:,} of {free_bytes / 1e9:.0f} GB free, largest chromosome {chromosome_cap:,} candidates in cap steps)")
    return cap


def _merged_certificate(certificates: Sequence[FitCertificate], groups: Sequence[I64Array]) -> FitCertificate:
    """One certificate over the request's models (one certificate each, in model order), fitted a group at a time:
    per-model terms stacked, whole-fit counts summed over the groups (each model of a group carries its group's, which
    are counted once), refusals joined with their model's index."""
    values = {}
    for field in dataclasses.fields(FitCertificate):
        parts = [getattr(certificate, field.name) for certificate in certificates]
        if field.name == "refusals":
            values[field.name] = tuple(f"model {model}: {reason}" for model, part in enumerate(parts) for reason in part)
        elif isinstance(parts[0], tuple):
            values[field.name] = tuple(entry for part in parts for entry in part)
        elif np.ndim(parts[0]) == 0:
            values[field.name] = sum(int(parts[int(group[0])]) for group in groups)
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
    annotations: F64Array | None = None, annotation_names: Sequence[str] = (), measurement: ColumnMeasurement | None = None,
) -> str:
    """The prior's schema, which its fitted coefficients do not carry: the lattice its density is written on, the
    variant class of each row the prior covers, those rows in the store, the offset each one was given, and the
    annotation design theta multiplies (its columns by name), and the covered rows' unit contract (raw unit, encoding
    scale, calibration slope, log reliability and its source: ``imputation_reliability.ColumnMeasurement``). Without it the saved hyperparameters name no density
    (``artifact.Provenance.prior_digest``). Empty arrays for a model with no prior at all (the null genetic model)."""
    return named_digest({
        "nodes": nodes, "floor": floor, "top": top, "class_index": class_index, "offsets": offsets, "rows": rows,
        "annotations": np.zeros((0, 0)) if annotations is None else annotations,
        "annotation_names": np.array(list(annotation_names), dtype=str),
        **({} if measurement is None else {
            "raw_unit": np.asarray(measurement.raw_unit[rows], dtype=str), "codes_per_unit": measurement.codes_per_unit[rows],
            "calibration_slope": measurement.calibration_slope[rows], "log_reliability": measurement.log_reliability[rows],
            "reliability_source": np.array([measurement.reliability_source], dtype=str),
        }),
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
    gram_pseudo_inverse: F64Array


def _covariate_least_squares(covariates: F64Array, targets: F64Array) -> _CovariateFit:
    alpha, _sums, rank, _singular = np.linalg.lstsq(covariates, targets, rcond=None)
    residual = targets - covariates @ alpha
    residual_sum = float(residual @ residual)
    degrees = int(targets.shape[0] - rank)
    resolution = max(covariates.shape) * _EPSILON * float(np.linalg.norm(targets))
    return _CovariateFit(
        gram_pseudo_inverse=np.linalg.pinv(covariates.T @ covariates),
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
        # No genetic draw moves the covariates: every draw's conditional mean is alpha, and their covariance is the
        # least squares' own, sigma^2 (C'C)^+.
        covariate_draws=np.repeat(covariate_fit.alpha[:, None], draw_count, axis=1),
        covariate_covariance=covariate_fit.noise * covariate_fit.gram_pseudo_inverse if np.isfinite(covariate_fit.noise) else np.zeros((covariate_fit.alpha.size,) * 2),
        gaussian_posterior=True,
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
        restored_best=np.zeros(1, dtype=bool),
        budget_unresolved=np.zeros(1, dtype=np.int64),
        mixture_components=np.zeros(1, dtype=np.int64),
        state_digest=state_digest([covariate_fit.alpha, np.array([covariate_fit.noise])])[None, :],
        far_field=np.zeros(1),
        far_field_scale=np.zeros(1),
        far_field_scale_error=np.zeros(1),
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


def _null_binary_model(covariates: F64Array, labels: F64Array, draw_count: int, reason: str) -> _ModelFit:
    """A binary training set with no genetic column to fit: the covariates alone on the Bernoulli bound
    (``binary_likelihood.fit_binary_covariates``: q(alpha) Gaussian at its Polya-Gamma fixed point), its predictive's
    intercept shift calibrated to the training prevalence. The certificate is the null genetic model's, with the sites'
    remaining gain as the likelihood's pending one."""
    ascent, alpha, covariance = fit_binary_covariates(covariates, labels, 0.5 / draw_count)
    log(f"stage2 wiring: null binary genetic model ({reason}); the covariates alone, bound {ascent.bound:.6g}")
    base = _null_genetic_model(
        _CovariateFit(alpha=alpha, noise=1.0, degrees=labels.shape[0] - covariates.shape[1], explained=False, gram_pseudo_inverse=covariance),
        draw_count, reason,
    )
    scoring = dataclasses.replace(
        base.scoring, trait_type=TraitType.BINARY, covariate_covariance=covariance,
        predictive_intercept_shift=calibrated_shift(ascent.state.predictor_mean, ascent.state.predictor_variance, labels),
    )
    certificate = dataclasses.replace(base.certificate, noise_gain=np.array([ascent.remaining_gain]))
    return dataclasses.replace(base, scoring=scoring, noise=1.0, certificate=certificate)


def _failed_group(covariates: F64Array, targets: F64Array, trait_type: TraitType, draw_count: int, reason: str) -> list[_ModelFit]:
    """A group whose fit raised (no certified fixed point on any route): each model as its covariates alone (the null
    genetic model's form, so the request still returns one fit per model), with ``reason`` among its refusals and its
    outer criterion unmet, which ``model_acceptance`` quarantines: it is saved, never scored or reported."""
    log(f"stage2 wiring: {reason}; each model is quarantined")
    fits = []
    for model in range(targets.shape[1]):
        if trait_type == TraitType.BINARY:
            null = _null_binary_model(covariates, targets[:, model], draw_count, reason)
        else:
            null = _null_genetic_model(_covariate_least_squares(covariates, targets[:, model]), draw_count, reason)
        fits.append(dataclasses.replace(null, certificate=_with_refusal(null.certificate, reason)))
    return fits


def _with_refusal(certificate: FitCertificate, reason: str) -> FitCertificate:
    """One model's ``certificate`` with ``reason`` appended to its refusals, its outer criterion unmet and one more
    unresolved trial."""
    return dataclasses.replace(
        certificate, refusals=(*certificate.refusals, reason),
        outer_criterion_met=np.zeros_like(certificate.outer_criterion_met, dtype=bool),
        unresolved=np.asarray(certificate.unresolved) + 1,
    )


def _model_certificate(certificate: FitCertificate, model: int, joint: str) -> FitCertificate:
    """Model ``model``'s certificate from its group's fit: its own entry of every per-model term, the fit's whole-fit
    counts, and the fit's refusals, each after ``joint``, which names the models fitted together (a refusal's "model k"
    is the k-th of them)."""
    values = {}
    for field in dataclasses.fields(FitCertificate):
        value = getattr(certificate, field.name)
        if field.name == "refusals":
            values[field.name] = tuple(f"{joint}{reason}" for reason in value)
        elif isinstance(value, tuple):
            values[field.name] = (value[model],)
        elif np.ndim(value) == 0:
            values[field.name] = value
        else:
            values[field.name] = np.asarray(value)[model : model + 1]
    return FitCertificate(**values)


def _covering_lattice(lattices: Sequence[tuple[F64Array, float, float]]) -> tuple[F64Array, float, float]:
    """One start lattice for a group's models from each one's (``stage0_lattice``): every one has the spacing
    ``derived_lattice`` sets from the member count and the tolerance, which the group's models share, so the
    lowest-starting one continued at that spacing past the highest end covers each one's nodes. Its floor is the lowest
    (below a model's floor every kernel is flat, so below a lower one too) and its top the highest; the lattice check
    then tests it at each model's fitted state. A single model's lattice is its own."""
    nodes = min((entry[0] for entry in lattices), key=lambda values: float(values[0]))
    spacing = float(nodes[1] - nodes[0])
    beyond = int(np.ceil((max(float(entry[0][-1]) for entry in lattices) - float(nodes[-1])) / spacing))
    return (
        np.concatenate([nodes, nodes[-1] + spacing * np.arange(1, beyond + 1)]),
        min(float(entry[1]) for entry in lattices),
        max(float(entry[2]) for entry in lattices),
    )


class _TargetLd:
    """Stage 0's LD store (``genotype_statistics.LdGramStore``) seen with only some of its targets: each block's
    projected scores are those targets' columns, and everything else is the store's own."""

    def __init__(self, ld: LdGramStore, targets: I64Array) -> None:
        self._ld = ld
        self._targets = targets
        self.target_count = int(targets.shape[0])

    def __getattr__(self, name: str) -> object:
        return getattr(self._ld, name)

    def block(self, block_index: int) -> ProjectedLdBlock:
        block = self._ld.block(block_index)
        return dataclasses.replace(block, projected_score=block.projected_score[:, self._targets])


def _target_view(statistics: GenotypeSufficientStatistics, targets: Sequence[int]) -> GenotypeSufficientStatistics:
    """Stage 0's statistics of a group (``_fit_group``) as those of its targets ``targets`` alone: the targets' own parts
    (C'y, Y'Y and each block's projected scores) selected, every other part, which no target enters, shared."""
    index = np.asarray(targets, dtype=np.int64)
    return dataclasses.replace(
        statistics, covariate_target=statistics.covariate_target[:, index], target_gram=statistics.target_gram[np.ix_(index, index)],
        ld=_TargetLd(statistics.ld, index),
    )


def _quarantine_codes(fit: _ModelFit) -> list[str]:
    """The fatal issues ``model_acceptance.acceptance`` finds in one model's fit: empty where it would be accepted."""
    certificate = fit.certificate
    terms = {
        field.name: np.asarray(getattr(certificate, field.name)) for field in dataclasses.fields(FitCertificate)
        if not isinstance(getattr(certificate, field.name), tuple) and np.ndim(getattr(certificate, field.name)) > 0
    }
    decision = acceptance(SimpleNamespace(certificate=terms, refusals=certificate.refusals, model_names=("model",), scoring=(fit.scoring,)))
    return [issue["code"] for issue in decision["issues"] if issue["severity"] == "fatal"]


def stage2_source(
    store: DosageStore, statistics: GenotypeSufficientStatistics, budget: ComputeBudget, workspace_bytes: int, training_columns: I64Array,
    code_cache: CodeCache | None = None, codes: TrainingCodes | None = None,
) -> StreamedDualSource | ShardedDualSource:
    """Stage 2's reads of the store on a model's training samples: sharded by LD block over every CUDA device of the
    budget where it has several (``device_plan.ShardedDualSource``: each block's products on its own device, a read's
    images summed on the first), else streamed on the one device or the host; on a device each block's codes are kept
    as a restart unit where ``code_cache`` names the store (``StoreGenotypeBlockSource._iter_prepared``). A read takes
    its blocks from their restart units where every block has one, else from ``codes``, the training codes Stage 0's
    pass kept (``training_codes``), while they are held, else from the store."""
    if budget.device_kind == "cuda" and len(budget.device_ids) > 1:
        return ShardedDualSource.from_statistics(
            store, statistics, budget, workspace_bytes, sample_indices=training_columns, code_cache=code_cache, codes=codes,
        )
    return StreamedDualSource(StoreGenotypeBlockSource.from_statistics(
        store, statistics, budget, workspace_bytes, sample_indices=training_columns, code_cache=code_cache, codes=codes,
    ))


def _fill_resident(source: StreamedDualSource | ShardedDualSource) -> None:
    """The read that fills the devices' resident genotype codes, made before the fit's first pass and on every device at
    once where Stage 2 is sharded (``ShardedDualSource.fill``; ``StoreGenotypeBlockSource.fill``). Nothing is read on
    the host, which holds none."""
    started = time.perf_counter()
    if source.fill() if isinstance(source, ShardedDualSource) else source.source.fill():
        log(f"stage2 wiring: the resident genotype codes are held on every device, filled in {time.perf_counter() - started:.0f} s")


def _stage0(
    store: DosageStore, candidates: I64Array, training_columns: I64Array, covariates: F64Array, targets: F64Array, config: ModelConfig,
    budget: ComputeBudget, work_dir: Path, restart: RestartStore, group: str, store_key: str,
    tiles: Callable[[int], DosageStoreTileSource] | None = None,
) -> tuple[Path, dict]:
    """Stage 0's restart unit for these inputs (``_fit_group``), made where there is none this machine can use. Its key
    holds the store's content, the candidate records, the training rows, covariates and targets in their order, the
    configuration, the devices it ran on (their kind, count and compute capabilities; the host's threads on the CPU),
    the source of Stage 0's modules with every module they import and of the unit's codec, never the engine's, and the
    block cap it was cut at. The cap is a resource choice (``_block_cap``: the memory's, the disk's and the largest
    chromosome's), so a unit of these inputs is found by its other facts and serves wherever this machine's memory and
    chromosomes allow its cap, the largest such first: recomputed from the disk a restored unit already fills, the cap
    fell and missed the unit at every restart. A unit cut at a cap this machine cannot hold is made again at this
    machine's, under its own key, which supersedes it here and, through the mirror, there (a whole mirrored unit is
    never replaced, so a rebuild under the same key would have been restored and refused at every restart). ``tiles``
    gives the pass's tile source at the cap it cuts (the EB monitor's tee, ``_fit_group``), which reads the same tiles."""
    named = facts({
        "store": store_key, "candidates": candidates, "training": training_columns, "covariates": covariates, "targets": targets,
        "config": repr(config), "devices": repr((budget.device_kind, len(budget.device_ids), budget.device_compute_capabilities,
                                                 budget.cpu_threads if budget.device_kind == "cpu" else None)),
        "source": module_sources("sv_pgs.genotype_statistics", "sv_pgs.dosage_store"),
        "codec": definition_sources([("sv_pgs.restart_store", "save_statistics"), ("sv_pgs.restart_store", "load_statistics")]),
    })
    usable = min(_cap_limits(store, candidates, training_columns, covariates.shape[1], budget))
    built = [
        (int(record["values"]["block_cap"]), path, record) for path, record in restart.units("stage0")
        if {name: value for name, value in record["facts"].items() if name != "block_cap"} == named
    ]
    serving = [entry for entry in built if entry[0] <= usable]
    if serving:
        cap, path, record = max(serving, key=lambda entry: entry[0])
        log(f"stage2 wiring: Stage 0 reused from its restart unit (block cap {cap:,})")
        return path, record
    block_cap = _block_cap(store, candidates, training_columns, covariates.shape[1], budget, work_dir)
    named = {**named, **facts({"block_cap": block_cap})}
    if built:
        log(f"stage2 wiring: Stage 0's restart units have block caps {', '.join(f'{cap:,}' for cap, _path, _record in built)}, above this machine's {usable:,}: made again")
    else:
        restart.explain("stage0", group, named)
    key = unit_key(named)
    with restart.build("stage0", key, group=group, named=named, values={"block_cap": block_cap}) as directory:
        statistics = compute_genotype_statistics(
            DosageStoreTileSource(store, candidates) if tiles is None else tiles(block_cap), training_columns, covariates, targets,
            config, budget, block_cap, directory / "ld",
        )
        save_statistics(directory, statistics)
        del statistics
    found = restart.unit("stage0", key)
    assert found is not None
    return found


def _band(
    ld: LdGramStore, sample_count: int, share: int, level: float, array_module, kappa: float, restart: RestartStore, group: str, stage0: str,
) -> BandStore:
    """The summary band's restart unit (``gram_space.build_band_store``), made where there is none: keyed by the Stage 0
    unit's build it is cut from, the samples, the test's level, kappa, the array module and the source of the band's
    builder and of the definitions it reaches, never of the sweeps beside it in ``gram_space``. The workspace is not in
    the key: it sets only the extent test's row chunks (``marginal_variances.ld_lag_profiles``), a summation order, and
    it is the free device memory each job measures, so as a key part it missed the band at every restart; it is
    recorded with the unit."""
    named = facts({
        "stage0": stage0, "samples": sample_count, "level": repr(float(level)), "kappa": repr(float(kappa)),
        "backend": array_module.__name__,
        "source": definition_sources([("sv_pgs.gram_space", "build_band_store"), ("sv_pgs.gram_space", "BandStore")]),
    })
    key = unit_key(named)
    found = restart.unit("band", key)
    if found is None:
        restart.explain("band", group, named)
        with restart.build("band", key, group=group, named=named, values={"share": share}) as directory:
            build_band_store(ld, sample_count, directory, share, level, array_module, kappa)
        found = restart.unit("band", key)
        assert found is not None
    else:
        log("stage2 wiring: summary band reused from its restart unit")
    return BandStore(found[0])


def _fitted_cavity(fit: FullDataFit, model: int) -> Cavity:
    """Model ``model``'s cavity at the fitted state: mean field's pseudo-likelihoods, or EP's own cavities."""
    if fit.member_omega is not None:
        return Cavity(precision=fit.member_omega[:, model], shift=fit.member_shift[:, model])
    return Cavity(precision=fit.cavity_precision[:, model], shift=fit.cavity_shift[:, model])


def _fit_group(
    store: DosageStore,
    training_columns: I64Array,
    covariates: F64Array,
    targets: F64Array,
    measurement: ColumnMeasurement,
    budget: ComputeBudget,
    work_dir: Path,
    seed: int,
    draw_count: int,
    models: I64Array,
    trait_type: TraitType = TraitType.QUANTITATIVE,
    inference: str = "mean_field",
    annotated_prior: bool = True,
    length_by_class: bool = False,
    ld_score_prior: bool = False,
    restart: RestartStore | None = None,
    group: str = "group",
    store_key: str | None = None,
    request_key: str | None = None,
    evaluation: GroupEvaluation | None = None,
) -> list[_ModelFit]:
    """The group of the request's ``models`` (``model_groups``) on the sorted store columns ``training_columns``, whose
    covariates (intercept first) and targets [n, models] follow them: one Stage 0 with a target column per model, one
    prior, one dual Gaussian with a column per model and one ``fit_full_data``, then each model's own lattice check,
    scoring, noise, hyperparameters, certificate and prior digest. A model is the null genetic model where its training
    set has no genetic column to fit.

    On the band route (``gram_space``, where a pass over the band measures cheaper) no model's fit is lost to the route: a
    group whose joint fit has no certified start is fitted model by model on the same Stage 0, and a model whose band-route
    fit has no certified start, or would be quarantined (``model_acceptance``: its outer criterion unmet, a returned state
    unresolved, a quadrature check left open, ...), is fitted again alone on the sample route, the reference route both
    compute; each such model's refusals say why.

    A binary model (0/1 targets) is a group of its own, fitted on its Polya-Gamma bound (``binary_likelihood``): after the
    separation check, Stage 0 runs on its start's equal-weight problem (every weight 1/4, response 4 kappa, noise 4), where
    the lattice and the EB start are exact, and the dual solver starts in that metric with its sites at xi = 0.

    With ``length_by_class`` (the hierarchical length-by-type prior) each variant class whose lengths vary bends the
    shared length smooth by its own learned deviation (``annotation_design``'s ``by_class``, ``BY_CLASS_ANNOTATIONS``);
    without it every class shares one length curve, the prior of the runs so far. With ``ld_score_prior`` the level of
    LD around each member, its Stage 0 LD score, is a learned smooth of the prior variance beside the frequency function
    (``annotation_design.ld_score_annotation``), in either prior form; without it the prior has none, as so far.

    ``evaluation``, the group's evaluation rows, is the EB monitor's alone (``eb_monitor``): Stage 0's pass keeps their
    codes as it reads, and the mean-field fit logs each outer-loop call's posterior mean scored on them. Nothing of the
    fit reads it. A Stage 0 reused from its restart unit reads no tile, so the monitor then logs its calls unscored.

    Its products are ``restart``'s units of ``group`` (``restart_store``; by default a store in ``work_dir``), each
    reused where its key matches: Stage 0 (keyed by ``store_key``, the candidates, the training rows, covariates and
    targets, the configuration, the devices and Stage 0's modules; its block cap a resource choice it records, reused
    where this machine's memory allows it), the far-field scale and the band (keyed by that Stage 0 unit's build and
    their own inputs and producing definitions), each block's codes (``CodeCache``) and each model's best mean-field
    start (``WarmStart``, keyed by ``request_key``). A ``store_key`` of None (a caller that did not digest the store)
    keys this call's products to it alone."""
    restart = restart if restart is not None else RestartStore(work_dir / "restart")
    store_key = store_key if store_key is not None else uuid.uuid4().hex
    request_key = request_key if request_key is not None else uuid.uuid4().hex
    # SPEC 1fca1cf: no variant is filtered by rarity or any threshold (only a derived bound may leave one out).
    config = ModelConfig(minimum_minor_allele_frequency=0.0)
    binary = trait_type == TraitType.BINARY
    labels = targets
    sites: list[BernoulliSites | None] = [None] * models.shape[0]
    if binary:
        for model in range(models.shape[0]):
            assert_no_separation(covariates, labels[:, model])
        sites = [BernoulliSites.start(labels[:, model]) for model in range(models.shape[0])]
        targets = np.column_stack([model_sites.response for model_sites in sites])
    # Before the store is read: what the covariates alone leave, which decides whether there is anything to fit.
    covariate_fits = [_covariate_least_squares(covariates, targets[:, model]) for model in range(models.shape[0])]
    if covariate_fits[0].degrees <= 0:
        raise ValueError(
            f"{targets.shape[0]} training rows against covariates of rank {targets.shape[0] - covariate_fits[0].degrees} leave no "
            "residual degrees of freedom, so the noise variance is not identified."
        )

    def null_model(model: int, reason: str) -> _ModelFit:
        if binary:
            return _null_binary_model(covariates, labels[:, model], draw_count, reason)
        return _null_genetic_model(covariate_fits[model], draw_count, reason)

    # (Never for a binary model: labels the covariates span exactly would separate, which the check refused.)
    results: list[_ModelFit | None] = [
        null_model(model, "the covariates explain every training target to working precision") if covariate_fit.explained else None
        for model, covariate_fit in enumerate(covariate_fits)
    ]
    # The models with a residual to fit, the columns of Stage 0's targets and of the dual solver.
    genetic = [model for model, result in enumerate(results) if result is None]

    def without_genetic_column(reason: str) -> list[_ModelFit]:
        return [null_model(model, reason) if result is None else result for model, result in enumerate(results)]

    if not genetic:
        return [result for result in results if result is not None]
    candidates = stage0_candidates(store, training_columns, measurement.log_reliability, config)
    if candidates.shape[0] == 0:
        return without_genetic_column("no store record carries signal on these training rows")
    count = len(genetic)
    store_targets = targets[:, genetic]
    model_sites = [sites[model] for model in genetic]
    # Stage 2's codes and the EB monitor's codes of the evaluation rows, kept from Stage 0's own tiles (``training_codes``)
    # where this job makes Stage 0 (``_stage0`` asks for its tiles at the cap it cuts them at): the resident codes and the
    # far field then read nothing from the store. Stage 2's are admitted first, so the ledger evicts the monitor's first.
    training_codes = None
    evaluation_codes = None

    def tiles(block_cap: int) -> DosageStoreTileSource:
        nonlocal training_codes, evaluation_codes
        if code_cache_on():
            try:
                training_codes = TrainingCodes(candidates, training_columns)
            except MemoryError as error:
                log(f"training codes: not kept: {error}")
        if evaluation is not None and evaluation.store_columns.size:
            try:
                evaluation_codes = EvaluationCodes(candidates, evaluation.store_columns, block_cap)
            except MemoryError as error:
                log(f"eb monitor: off: {error}")
        keepers = [codes for codes in (training_codes, evaluation_codes) if codes is not None]
        return CodeTee(store, candidates, keepers) if keepers else DosageStoreTileSource(store, candidates)

    stage0_unit = _stage0(store, candidates, training_columns, covariates, store_targets, config, budget, work_dir, restart, group, store_key, tiles)
    statistics = load_statistics(stage0_unit[0])
    block_cap = statistics.boundaries.block_cap
    if np.asarray(statistics.active_rows).shape[0] == 0:
        for codes in (training_codes, evaluation_codes):
            if codes is not None:
                codes.release()
        return without_genetic_column("every candidate record is monomorphic on these training rows")
    kept_rows = np.asarray(statistics.active_rows, dtype=np.int64)[np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)]
    # The prior is over every active row: tie members keep their own class and offset (tie_members; review-mathbugs T1).
    member_rows = np.asarray(statistics.active_rows, dtype=np.int64)
    monitor = None if evaluation is None else EbMonitor.start(
        evaluation_codes, member_rows, np.asarray(statistics.means), np.asarray(statistics.scales), evaluation,
    )
    # The unit contract (``imputation_reliability.ColumnMeasurement``): each member's offset log(kappa^2 Var(D)) in its
    # stored value's raw units (code sd over codes_per_unit; a copy number's differ), with kappa = 1 because the store
    # holds calibrated conditional means (a GLIMPSE2/Beagle DS, or the measurement step's rewritten D*), and its log
    # true-genotype variance, the frequency function's argument, where the reliability enters.
    offsets, log_genotype_variance = measurement.standardized_terms(member_rows, np.asarray(statistics.scales))
    classes, class_index = np.unique(store.variant_table.variant_class[member_rows], return_inverse=True)
    class_labels = [VARIANT_CLASSES[int(code)].value for code in classes]
    table = store.variant_table
    # The store's annotations per member and the frequency function; ``fit_full_data`` nests them on the prior without
    # annotations, so each enters where the evidence rises.
    # The simpler prior (``annotated_prior`` False, the 2 x 2 comparison's) keeps the frequency function and drops the
    # store's annotations.
    member_annotations = {name: np.asarray(values)[member_rows] for name, values in table.annotations.items()} if annotated_prior else {}
    member_annotations.update(frequency_annotation(log_genotype_variance))
    if ld_score_prior:
        # Each member's reduced column's LD score, which an exact tie shares (the prior is over every member).
        member_annotations.update(ld_score_annotation(
            np.asarray(statistics.ld.ld_scores())[np.asarray(statistics.tie_map.original_to_reduced, dtype=np.int64)]
        ))
    annotations = annotation_design(
        member_annotations,
        table.annotation_legends,
        class_index=class_index.astype(np.int64),
        exclude=RELIABILITY_COLUMNS + IDENTIFIER_COLUMNS,
        by_class=BY_CLASS_ANNOTATIONS if length_by_class else (),
        class_labels=class_labels,
    )
    log(f"stage2 wiring: {annotations.design.shape[1]} annotation columns in {len(annotations.groups)} groups: {', '.join(annotations.names) or 'none'}")
    # These are the group's models: rows outside its training set have zero likelihood
    # weight. Keep only training rows in every iterative genotype product and
    # sample-space buffer. Scoring still reads the original full store.
    fitted_samples = training_columns.size
    mask = np.ones((fitted_samples, count))
    # A binary model's start is the equal-weight problem, whose noise is 1 / omega(0) = 4 exactly, in the response's units.
    start_noise = np.full(count, 1.0 / POLYA_GAMMA_MEAN_AT_ZERO) if binary else covariate_residual_variance(store_targets, mask, covariates)
    tolerance = 0.5 / draw_count
    nodes, floor, top = _covering_lattice([stage0_lattice(statistics, target, float(start_noise[target]), offsets, tolerance) for target in range(count)])
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
    source = stage2_source(store, statistics, budget, share, training_columns, CodeCache(restart, group, store_key), training_codes)
    # The devices' resident codes filled now, from their restart units or Stage 0's kept codes where there are: the
    # fit's first pass, the band's pass costs among them, is then a device pass.
    _fill_resident(source)
    # Timing may select a band accelerator, never a different likelihood. Every route
    # retains DualGaussian and recomputes the omitted fields from the full sample operator.
    band = None
    kappa = 1.0
    if not binary:
        started = time.perf_counter()
        # kappa first, on Stage 0's blocks (pairs beyond a block's neighbours): the extent test reads each block's LD
        # against kappa / n, and the band's far field is scaled by it.
        ld = statistics.ld
        stage0_starts = np.asarray(ld.block_boundaries, dtype=np.int64)
        stage0_linked = np.array([ld.has_adjacent(block + 1) for block in range(ld.block_count - 1)], dtype=bool)
        far_named = facts({
            "stage0": stage0_unit[1]["id"], "store": store_key, "training": training_columns, "covariates": covariates, "draws": draw_count,
            "seed": str(_seed(seed, _FAR_FIELD_KEY)), "source": definition_sources([("sv_pgs.gram_space", "far_field_scale")]),
        })
        far_unit = restart.unit("farfield", unit_key(far_named))
        if far_unit is None:
            restart.explain("farfield", group, far_named)
            measured = far_field_scale(
                store, training_columns, covariates, statistics, stage0_starts, stage0_linked, draw_count, _seed(seed, _FAR_FIELD_KEY),
                codes=training_codes,
            )
            with restart.build("farfield", unit_key(far_named), group=group, named=far_named) as directory:
                (directory / "scale.json").write_text(json.dumps(list(measured)), encoding="utf-8")
            far_unit = restart.unit("farfield", unit_key(far_named))
        else:
            log("stage2 wiring: far field: reused its restart unit")
        kappa, kappa_error, pairs = json.loads((far_unit[0] / "scale.json").read_text(encoding="utf-8"))
        log(f"stage2 wiring: far field: mean r^2 n' = {kappa:.4g} +- {kappa_error:.2g} over {pairs:,} pairs beyond Stage 0's neighbouring blocks (chance alone: 1)")
    if not binary and inference == "mean_field":
        # The summary band is the mean-field route's; EP runs over the samples, its windows cut at the LD extent read
        # against the same kappa (``null_scale``).
        band_store = _band(ld, statistics.sample_count, share, certificate_level(draw_count), source.array_module, kappa, restart, group, stage0_unit[1]["id"])
        band = GramBand(statistics, share, store=band_store)
        band.far_scale = kappa
        sample_seconds, gram_seconds = pass_costs(band, source, source.array_module)
        log(
            f"stage2 wiring: a pass costs {sample_seconds:.3g} s over the samples and {gram_seconds:.3g} s over the summary band "
            f"({band.group_count:,} columns, {band.block_count:,} parts, band {band.band_width:,}; built in {time.perf_counter() - started:.0f} s): "
            f"{'band-accelerated full-data' if gram_seconds < sample_seconds else 'full-data'} route"
        )
        if not gram_seconds < sample_seconds:
            band.release()
            band = None
    # The columns of Stage 0 (positions in ``genetic``) whose fits ran on the band route, and the notes each column's
    # fit was given.
    banded: set[int] = set()
    noted: dict[int, tuple[str, ...]] = {}

    def fitted(columns: list[int], on_band: bool, notes: tuple[str, ...]) -> None:
        """The models ``columns`` (positions in ``genetic``) fitted together on one dual Gaussian with a column each, on
        the band route where ``on_band``, each with ``notes`` (why it is fitted apart from its group, or on the samples)
        among its refusals; raises ``FloatingPointError`` where the fit has no certified start."""
        view = statistics if len(columns) == count else _target_view(statistics, columns)
        fit_band = None
        if on_band:
            fit_band = band if len(columns) == count else GramBand(view, share, store=band_store)
            fit_band.far_scale = kappa
        fit_sites = [model_sites[column] for column in columns]
        gaussian = DualGaussian(
            source=source,
            training=mask[:, columns],
            targets=store_targets[:, columns],
            offsets=np.zeros((fitted_samples, len(columns))),
            covariates=covariates,
            # The dual solver reads the Grams' blocks and links only, never their metric's scale.
            grams=block_grams(statistics),
            # The probes estimate the variance map's traces, which only the EP route reads: the mean-field route's
            # responses are the dual solver's solves, so its iterates carry no probe columns (each solved to
            # sqrt(eps) relative, draw_count of them per model, at every fixed point).
            probe_count=draw_count if inference == "ep" else 0,
            seed=_seed(seed, 0),
            probe_sample_indices=training_columns if fitted_samples != store.n_samples else None,
            # A binary model's metric at its start sites: W = omega(0) on the training rows, at unit noise.
            sample_weights=np.column_stack([entry.weights for entry in fit_sites]) if binary else None,
        )
        # A refusal of the joint fit names the models fitted together; its "model k" is the k-th of them.
        joint = f"joint fit of models {', '.join(str(int(models[genetic[column]])) for column in columns)}: " if len(columns) > 1 else ""

        def finish(fit: FullDataFit, fitted_prior: ScaleMixturePrior, finished: Sequence[int], open_checks: dict[int, str]) -> None:
            """The models ``finished`` from ``fit`` on ``fitted_prior``'s lattice: their scoring models, noise,
            hyperparameters, certificates (with the lattice check each one's fit left open) and prior digest."""
            scoring = scoring_models(fit, fitted_prior, view, [trait_type] * len(columns), draw_count, seed=_seed(seed, 2))
            digest = _prior_schema_digest(
                nodes=fitted_prior.log_variance_grid, floor=np.array([fitted_prior.kernel_floor]), top=np.array([fitted_prior.kernel_top]),
                class_index=class_index.astype(np.int64), offsets=offsets, rows=member_rows, annotations=annotations.design,
                annotation_names=annotations.names, measurement=measurement,
            )
            for model in finished:
                certificate = _model_certificate(fit.certificate, model, joint)
                certificate = dataclasses.replace(certificate, refusals=(*certificate.refusals, *notes))
                if model in open_checks:
                    certificate = _with_refusal(certificate, open_checks[model])
                    log(f"stage2 wiring: model {int(models[genetic[columns[model]]])}: {open_checks[model]}")
                results[genetic[columns[model]]] = _ModelFit(
                    scoring=scoring[model], noise=float(np.asarray(fit.noise_variance)[model]), hyperparameters=fit.hyperparameters[model],
                    certificate=certificate, prior_digest=digest,
                )
                # The learned length curves (per class where ``length_by_class`` gave the classes their own), the
                # prior's answer to how a structural variant's effect variance grows with its length.
                for name in LOG_SCALE_ANNOTATIONS:
                    record = smooth_shift_record(
                        fitted_prior.scale_design, fitted_prior.annotation_groups,
                        np.asarray(fit.hyperparameters[model].coefficients)[fitted_prior.coefficient_size - fitted_prior.scale_size :], name,
                        class_index, member_annotations[name], class_labels,
                    ) if name in member_annotations else ""
                    if record:
                        log(f"stage2 wiring: model {int(models[genetic[columns[model]]])}: {name}'s shift of the log prior variance by decade, per class (mean zero within a class): {record}")

        # The EB monitor's view of these models (their positions in the group).
        watched = None if monitor is None else monitor.view([genetic[column] for column in columns])

        def solve(fit_prior: ScaleMixturePrior, **continuation) -> FullDataFit:
            fit = fit_full_data(
                gaussian=gaussian, statistics=view, prior=fit_prior, draw_count=draw_count, working_bytes=share, seed=_seed(seed, 1),
                inference=inference, sites=fit_sites, null_scale=kappa, band=fit_band, monitor=watched,
                warm_start=WarmStart(restart, group, request_key), **continuation,
            )
            if fit_band is not None and np.any(fit.certificate.outer_criterion_met):
                # The band's far-field stopping estimate can miss a slowly converging direction: a grouped fixture
                # reported convergence but its held-out scores correlated only 0.951 with the sample-route reference.
                # Continue from its fitted state on the samples before accepting it. An unresolved continuation is
                # handled by the existing per-model sample-route fallback, rather than accepting the band's certificate.
                log(f"stage2 wiring: continuing the band fit of {len(columns)} models on the sample route for final validation")
                fit_band.release()
                fit = fit_full_data(
                    gaussian=gaussian, statistics=view, prior=fit_prior, draw_count=draw_count, working_bytes=share, seed=_seed(seed, 1),
                    inference=inference, sites=fit_sites, null_scale=kappa, band=None, starts=fit.hyperparameters,
                    start_noise=np.asarray(fit.noise_variance, dtype=np.float64), start_mean=fit.member_mean, monitor=watched,
                    warm_start=WarmStart(restart, group, request_key),
                )
            return fit

        # ``inference``'s fixed points: the mean-field route by default until the EP route (with its double loop, which
        # passes the wiring store EP refused 59 of 65 calls on) is measured on bench-sim.
        # The quadrature at the fitted state (review F17, ``scale_mixture_ep.lattice_check``): the start lattice is
        # derived from the start's single-variant likelihoods only. Each model whose check passes is finished from the fit
        # as it stands; where any model's fitted normalizers or moments move on the refined or the extended lattice, the
        # models are fitted again on the lattice moved for them, and those are checked there.
        fit_prior = prior
        try:
            fit = solve(fit_prior)
            pending = list(range(len(columns)))
            while pending:
                # Completed solver closures can form cycles holding large NumPy buffers. Their byte size does not
                # trigger Python's allocation-count GC threshold; release unreachable states before a larger lattice.
                gc.collect()
                checks = {model: lattice_check(fit_prior, fit.hyperparameters[model], _fitted_cavity(fit, model), share, draw_count) for model in pending}
                for model, check in checks.items():
                    log(f"stage2 wiring: lattice check of model {int(models[genetic[columns[model]]])} on {fit_prior.grid_size} nodes: {check.record()}")
                moving = [model for model in pending if checks[model].refine or checks[model].extend]
                passed = [model for model in pending if model not in moving]
                if passed:
                    finish(fit, fit_prior, passed, {})
                if not moving:
                    break
                moved, moved_hyperparameters = fit_prior, list(fit.hyperparameters)
                if any(checks[model].extend for model in moving):
                    moved, moved_hyperparameters = _move_lattice(moved, moved_hyperparameters, extended_lattice)
                if any(checks[model].refine for model in moving):
                    moved, moved_hyperparameters = _move_lattice(moved, moved_hyperparameters, halved_lattice)
                try:
                    # Continued from the fit on the checked lattice (its hyperparameters carried to the moved lattice, its
                    # noise and its members' means), not from the prior: on bench-sim chr22 [sim] the refit from the prior
                    # repeated the whole nested search, 50 minutes of the fit, to reach the state the first fit had.
                    refit = solve(
                        moved, starts=moved_hyperparameters, start_noise=np.asarray(fit.noise_variance, dtype=np.float64), start_mean=fit.member_mean,
                    )
                except FloatingPointError as error:
                    # No fit on the checked lattice: these models keep the fit they have, and each certificate says the
                    # quadrature check that fit failed was never resolved, so the artifact never reads as a passed check.
                    finish(fit, fit_prior, moving, {
                        model: f"lattice check unresolved ({checks[model].record()}): the fit on {fit_prior.grid_size} nodes stands; the refit on {moved.grid_size} nodes failed: {error}"
                        for model in moving
                    })
                    break
                # A model whose refit ended without its outer criterion (an uncertified walk) where the fit it continued
                # had met it keeps the certified fit, and its certificate records the quadrature check left open.
                kept = [
                    model for model in moving
                    if fit.certificate.outer_criterion_met[model] and not refit.certificate.outer_criterion_met[model]
                ]
                if kept:
                    finish(fit, fit_prior, kept, {
                        model: f"lattice check unresolved ({checks[model].record()}): the refit on {moved.grid_size} nodes did not meet its outer criterion; the fit on {fit_prior.grid_size} nodes stands"
                        for model in kept
                    })
                pending = [model for model in moving if model not in kept]
                fit_prior, fit = moved, refit
        finally:
            if fit_band is not None and fit_band is not band:
                fit_band.release()
        if on_band:
            banded.update(columns)
        noted.update({column: notes for column in columns})

    everyone = list(range(count))
    if band is None:
        fitted(everyone, False, ())
    else:
        try:
            fitted(everyone, True, ())
        except FloatingPointError as error:
            # Nothing after this reads the group's band (a model's own band-route fit has its own band, and the gate's
            # refits run on the samples): its cache goes before the models' fits build theirs.
            band.release()
            if count == 1:
                log(f"stage2 wiring: model {int(models[genetic[0]])}: the band route's fit has no certified start ({error}); fitted again on the sample route")
                fitted(everyone, False, (f"the band route's fit had no certified start, so it was fitted on the sample route: {error}",))
            else:
                note = f"the band route's joint fit of models {', '.join(str(int(models[model])) for model in genetic)} had no certified start, so each was fitted on its own: {error}"
                log(f"stage2 wiring: {note}")
                for column in everyone:
                    try:
                        fitted([column], True, (note,))
                    except FloatingPointError as single:
                        log(f"stage2 wiring: model {int(models[genetic[column]])}: the band route's fit alone has no certified start ({single}); fitted again on the sample route")
                        fitted([column], False, (note, f"alone on the band route it had no certified start either, so it was fitted on the sample route: {single}"))
        # The band route answers for a fit only where the release gate would accept it: a model it would quarantine is
        # fitted again alone on the sample route, the reference route both compute.
        for column in sorted(banded):
            codes = _quarantine_codes(results[genetic[column]])
            if codes:
                log(f"stage2 wiring: model {int(models[genetic[column]])}: its band-route fit would be quarantined ({', '.join(codes)}); fitted again alone on the sample route")
                try:
                    fitted([column], False, (*noted[column], f"the band route's fit would be quarantined ({', '.join(codes)}), so it was fitted again alone on the sample route"))
                except FloatingPointError as error:
                    # No sample-route fit either: the band-route fit stands, its fatal terms still quarantine this model
                    # alone, its refusals say the refit failed, and the group's other models keep their fits.
                    kept = results[genetic[column]]
                    log(f"stage2 wiring: model {int(models[genetic[column]])}: its sample-route refit has no certified start either ({error}); its band-route fit stands")
                    results[genetic[column]] = dataclasses.replace(kept, certificate=dataclasses.replace(kept.certificate, refusals=(
                        *kept.certificate.refusals,
                        f"the band route's fit would be quarantined ({', '.join(codes)}) and its sample-route refit had no certified start, so it stands: {error}",
                    )))
        band.release()
    if monitor is not None:
        monitor.close()
    if training_codes is not None:
        # Held to here: a device's resident set that a later peak evicts is filled again from its restart units or
        # from them, not from the store.
        training_codes.release()
    log(f"stage2 wiring: {kept_rows.shape[0]:,} reduced columns in {statistics.ld.block_count} blocks (cap {block_cap}), {training_columns.shape[0]:,} training samples, {count} models fitted together")
    return [result for result in results if result is not None]


def model_groups(store_columns: I64Array, covariate_columns: BoolArray, training: BoolArray, trait_types: Sequence[TraitType]) -> list[I64Array]:
    """The request's models as the groups fitted together (``_fit_group``), each in model order and the groups in the
    order of their first models: the models whose training rows (as store columns), adjusted covariates and trait type
    are the same. Stage 0's Grams and band, the candidates and the prior's schema read the rows, the covariates and the
    records, never a target, so these models share them and the dual solver's passes. A binary model is a group of its
    own: its metric moves with its own Polya-Gamma sites."""
    columns = np.asarray(store_columns, dtype=np.int64)
    groups: dict[tuple, list[int]] = {}
    for model, trait_type in enumerate(trait_types):
        key = (
            np.sort(columns[np.asarray(training[:, model], dtype=bool)]).tobytes(),
            np.asarray(covariate_columns[model], dtype=bool).tobytes(),
            trait_type,
            model if trait_type == TraitType.BINARY else None,
        )
        groups.setdefault(key, []).append(model)
    return [np.array(models, dtype=np.int64) for models in groups.values()]


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
    inference: str = "mean_field",
    checkpoint_key: str | None = None,
    annotated_prior: bool = True,
    evaluation: EvaluationRows | None = None,
    model_names: Sequence[str] | None = None,
    length_by_class: bool = False,
    ld_score_prior: bool = False,
    restart: RestartStore | None = None,
    identity: Mapping[str, str] | None = None,
    store_key: str | None = None,
    request_key: str | None = None,
) -> FittedModels:
    """Every model of ``fit_model.fit``, a group at a time (``model_groups``; see the module docstring); ``inference``
    names the fixed point (``full_data_fit.fit_full_data``): the mean-field route until the EP route (structured-genome's
    genome-scale EP) is measured on bench-sim, which then replaces it.

    ``evaluation`` (its covariates with the intercept first, as ``covariates``) gives each model's evaluation rows to
    the EB monitor alone (``eb_monitor``), which logs each outer-loop call's held-out metrics under ``model_names``;
    nothing fitted, saved or checkpointed reads it. With the monitor switched off (``eb_monitor.MONITOR_SWITCH``) it
    is dropped here, before any group.

    Each model's completed fit is checkpointed on its own under ``checkpoint_key`` in ``restart`` (by default a store in
    ``work_dir``: checkpoints/<key>/model<i>.npz), a unit whose facts are ``identity`` (``fit_model.fit_identity``'s named
    digests), so a fit whose checkpoints no longer match logs which named input changed; a group is fitted again unless
    every one of its models is, and a group's seed is the request's with its models' indices, so a rerun fits it the same
    way. A group's other units (Stage 0, the band, its codes and EB starts: ``_fit_group``) are released once its models
    are checkpointed."""
    if any(trait_type not in (TraitType.QUANTITATIVE, TraitType.BINARY) for trait_type in trait_types):
        raise ValueError(f"unknown trait types in {list(trait_types)}")
    # Stage 0 runs a buffer on every device, and Stage 2's reads are sharded over them by LD block (``stage2_source``).
    stage2_devices = f"CUDA devices {', '.join(str(device) for device in budget.device_ids)}" if budget.device_kind == "cuda" else "the CPU"
    log(f"stage2 wiring: Stage 0 and Stage 2's reads run on {stage2_devices}")
    restart = restart if restart is not None else RestartStore(work_dir)
    restart.clean()
    if checkpoint_key is not None:
        named = dict(identity) if identity is not None else {"checkpoint": str(checkpoint_key)}
        if restart.unit("checkpoints", checkpoint_key) is None:
            restart.explain("checkpoints", REQUEST, named)
        # One key's checkpoints a request: another key's, which no fit of this request reads, are released here, the
        # mirror's at its next push.
        restart.mutable("checkpoints", checkpoint_key, group=REQUEST, named=named, supersede=True)
    measurement = store_measurement(store, log_variance_offset)
    if evaluation is not None and not monitor_on():
        log(f"eb monitor: off ({MONITOR_SWITCH}=0): no codes kept, nothing scored")
        evaluation = None
    groups = model_groups(store_columns, covariate_columns, training, trait_types)
    log(f"stage2 wiring: {training.shape[1]} models in {len(groups)} groups fitted together (models sharing training rows, covariates and likelihood)")
    fits: list[_ModelFit | None] = [None] * training.shape[1]
    for index, group in enumerate(groups):
        first = int(group[0])
        rows = np.flatnonzero(training[:, first])
        order = np.argsort(store_columns[rows], kind="stable")
        rows = rows[order]
        adjusted = np.asarray(covariate_columns[first], dtype=bool)
        label = ", ".join(str(int(model)) for model in group)
        group_dir = Path(work_dir) / f"model{first}"
        checkpoints = [restart.directory / "checkpoints" / str(checkpoint_key) / f"model{int(model)}.npz" for model in group]
        if checkpoint_key is not None and all(checkpoint.is_file() for checkpoint in checkpoints):
            group_fits = [load_checkpoint(checkpoint, checkpoint_key) for checkpoint in checkpoints]
            log(f"stage2 wiring: resumed completed models {label}")
        else:
            # Incomplete scratch is disposable; completed models live outside this directory.
            if group_dir.exists():
                _remove_group_scratch(group_dir)
            group_dir.mkdir(parents=True)
            log(f"stage2 wiring: group {index + 1} of {len(groups)}: models {label} on {rows.shape[0]:,} training rows")
            started = time.perf_counter()
            group_covariates = np.asarray(covariates[rows][:, adjusted], dtype=np.float64)
            group_targets = np.asarray(targets[np.ix_(rows, group)], dtype=np.float64)
            names = [str(int(model)) if model_names is None else model_names[int(model)] for model in group]
            group_evaluation = None if evaluation is None else evaluation.group(group, adjusted, names)
            # One ledger per group's fit: its host leases and every device allocation.
            try:
                with memory_scope(budget):
                    group_fits = _fit_group(
                        store,
                        np.asarray(store_columns[rows], dtype=np.int64),
                        group_covariates,
                        group_targets,
                        measurement,
                        budget,
                        group_dir,
                        _seed(seed, *group.tolist()),
                        draw_count,
                        group,
                        trait_types[first],
                        inference,
                        annotated_prior,
                        length_by_class,
                        ld_score_prior,
                        restart,
                        group_dir.name,
                        store_key,
                        request_key,
                        group_evaluation,
                    )
            except FloatingPointError as error:
                # A fit with no certified fixed point on any route is the same failure on every rerun (its seeds are
                # its own), so it costs its models, not the request: each is quarantined (``_failed_group``) and
                # checkpointed like any finished one, and every other group is fitted.
                group_fits = _failed_group(group_covariates, group_targets, trait_types[first], draw_count, f"models {label}: the fit failed: {error}")
            elapsed = time.perf_counter() - started
            if checkpoint_key is not None:
                for checkpoint, fit in zip(checkpoints, group_fits):
                    # Each model's share of its group's time, so the shares sum to the fit's.
                    save_checkpoint(checkpoint, fit, checkpoint_key, elapsed / group.shape[0])
            log(f"stage2 wiring: completed models {label} in {elapsed:.3f} seconds")
        # Its models are checkpointed: no restart reads the group's Stage 0, band, codes or EB starts again.
        restart.release(group_dir.name)
        # The group's Stage 0 LD blocks and band (its projected Grams, O(p x block cap) bytes on disk) are read by no
        # later group and not by scoring, which the fit already ran: each is removed as its group ends, so the work
        # directory never holds more than one group's.
        if group_dir.exists():
            _remove_group_scratch(group_dir)
        for model, fit in zip(group.tolist(), group_fits):
            alpha = np.zeros(adjusted.shape[0])
            alpha[adjusted] = fit.scoring.alpha
            covariate_draws = np.zeros((adjusted.size, fit.scoring.draw_count))
            covariate_draws[adjusted] = fit.scoring.covariate_draws
            covariate_covariance = np.zeros((adjusted.size, adjusted.size))
            covariate_covariance[np.ix_(adjusted, adjusted)] = fit.scoring.covariate_covariance
            fits[model] = dataclasses.replace(
                fit, scoring=dataclasses.replace(fit.scoring, alpha=alpha, covariate_draws=covariate_draws, covariate_covariance=covariate_covariance),
            )
    finished = [fit for fit in fits if fit is not None]
    return FittedModels(
        scoring=[fit.scoring for fit in finished],
        noise_variance=np.array([fit.noise for fit in finished]),
        hyperparameters=tuple(fit.hyperparameters for fit in finished),
        certificate=_merged_certificate([fit.certificate for fit in finished], groups),
        prior_digests=tuple(fit.prior_digest for fit in finished),
    )

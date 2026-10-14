"""Store -> fitted models: the production wiring of Stage 0, the prior and Stage 2, run once per group of models.

The models that train on the same rows with the same covariates and the same likelihood are one group
(``model_groups``): Stage 0's Grams, its LD blocks and band, the candidate records and the prior's schema depend on the
rows, the covariates and the records, never on a target, so the group's models share them and the dual solver's passes
(docs/design/COMPUTE.md: each further model adds right-hand-side columns, not passes). Each group gets:
Stage 0 on its training rows and covariate columns with every model's target as a column, the start lattice covering its
models' single-variant likelihoods, the prior with one class per variant class present, the records' unit offsets and
frequency function from the store's unit contract (``store_measurement``), the store's other sidecar columns as its
annotation groups (``annotation_design``), the dual Gaussian with a column per model, ``fit_full_data`` by the
mean-field fixed points (``full_data_fit._FullDataMeanField``) and ``scoring_models``. Each model keeps its own
hyperparameters, noise, lattice check, certificate and prior digest, as when fitted alone: a group shares the store's
reads, never a hyperparameter, so there is no cross-trait pooling of the prior. A group's models share its probes' and
draws' seed, so their results are those of separate fits up to that Monte Carlo error.
A binary model (0/1 targets) is a group of its own, fitted on its Polya-Gamma bound (``binary_likelihood``), on the same
route; its reported noise variance is its whitened problem's known unit noise, which its predictive never reads.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

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
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.gram_space import GramBand, build_band_store, far_field_scale, pass_costs
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
from sv_pgs.annotation_design import annotation_design, frequency_annotation
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import Cavity, MixtureHyperparameters, ScaleMixturePrior, extended_lattice, halved_lattice, lattice_check, scale_mixture_prior
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
    return ColumnMeasurement.build(
        store.n_variants, log_reliability=reliability, codes_per_unit=table.codes_per_unit,
        raw_unit=np.where(copy_number, "copy", "ALT allele").astype(object), reliability_source=source,
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
    sample_groups = np.full(store.n_samples, -1, dtype=np.int64)
    sample_groups[training_columns] = 0
    memory_cap = stage0_block_cap(budget, build_sample_layout(sample_groups), covariate_count + 1)
    free_bytes = shutil.disk_usage(work_dir).free
    try:
        disk_cap = largest_block_cap(free_bytes // 2, lambda cap: int(candidates.shape[0]) * cap * LD_STORE_BYTES_PER_VARIANT_AND_CAP)
    except MemoryError as error:
        raise OSError(f"{work_dir}'s {free_bytes / 1e9:.1f} GB free cannot hold Stage 0's LD store and band for {candidates.shape[0]:,} candidates: {error}") from error
    chromosome_of_row = np.searchsorted(np.asarray(store.chromosome_starts), candidates, side="right") - 1
    largest = int(np.bincount(chromosome_of_row).max())
    cap = min(memory_cap, disk_cap, -(-largest // BLOCK_CAP_STEP) * BLOCK_CAP_STEP)
    log(f"stage2 wiring: block cap {cap:,} (memory {memory_cap:,}, disk {disk_cap:,} of {free_bytes / 1e9:.0f} GB free, largest chromosome {largest:,} candidates)")
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
) -> StreamedDualSource | ShardedDualSource:
    """Stage 2's reads of the store on a model's training samples: sharded by LD block over every CUDA device of the
    budget where it has several (``device_plan.ShardedDualSource``: each block's products on its own device, a read's
    images summed on the first), else streamed on the one device or the host."""
    if budget.device_kind == "cuda" and len(budget.device_ids) > 1:
        return ShardedDualSource.from_statistics(store, statistics, budget, workspace_bytes, sample_indices=training_columns)
    return StreamedDualSource(StoreGenotypeBlockSource.from_statistics(store, statistics, budget, workspace_bytes, sample_indices=training_columns))


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
    the lattice and the EB start are exact, and the dual solver starts in that metric with its sites at xi = 0."""
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
    block_cap = _block_cap(store, candidates, training_columns, covariates.shape[1], budget, work_dir)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, candidates), training_columns, covariates, store_targets, config, budget, block_cap, work_dir / "ld"
    )
    if np.asarray(statistics.active_rows).shape[0] == 0:
        return without_genetic_column("every candidate record is monomorphic on these training rows")
    kept_rows = np.asarray(statistics.active_rows, dtype=np.int64)[np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)]
    # The prior is over every active row: tie members keep their own class and offset (tie_members; review-mathbugs T1).
    member_rows = np.asarray(statistics.active_rows, dtype=np.int64)
    # The unit contract (``imputation_reliability.ColumnMeasurement``): each member's offset log(kappa^2 Var(D)) in its
    # stored value's raw units (code sd over codes_per_unit; a copy number's differ), with kappa = 1 because the store
    # holds calibrated conditional means (a GLIMPSE2/Beagle DS, or the measurement step's rewritten D*), and its log
    # true-genotype variance, the frequency function's argument, where the reliability enters.
    offsets, log_genotype_variance = measurement.standardized_terms(member_rows, np.asarray(statistics.scales))
    _classes, class_index = np.unique(store.variant_table.variant_class[member_rows], return_inverse=True)
    table = store.variant_table
    # The store's annotations per member and the frequency function; ``fit_full_data`` nests them on the prior without
    # annotations, so each enters where the evidence rises.
    # The simpler prior (``annotated_prior`` False, the 2 x 2 comparison's) keeps the frequency function and drops the
    # store's annotations.
    member_annotations = {name: np.asarray(values)[member_rows] for name, values in table.annotations.items()} if annotated_prior else {}
    member_annotations.update(frequency_annotation(log_genotype_variance))
    annotations = annotation_design(
        member_annotations,
        table.annotation_legends,
        class_index=class_index.astype(np.int64),
        exclude=RELIABILITY_COLUMNS + IDENTIFIER_COLUMNS,
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
    source = stage2_source(store, statistics, budget, share, training_columns)
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
        kappa, kappa_error, pairs = far_field_scale(
            store, training_columns, covariates, statistics, stage0_starts, stage0_linked, draw_count, _seed(seed, _FAR_FIELD_KEY),
        )
        log(f"stage2 wiring: far field: mean r^2 n' = {kappa:.4g} +- {kappa_error:.2g} over {pairs:,} pairs beyond Stage 0's neighbouring blocks (chance alone: 1)")
    if not binary and inference == "mean_field":
        # The summary band is the mean-field route's; EP runs over the samples, its windows cut at the LD extent read
        # against the same kappa (``null_scale``).
        band_store = build_band_store(ld, statistics.sample_count, work_dir / "band", share, certificate_level(draw_count), source.array_module, kappa)
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
            probe_count=draw_count,
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

        # ``inference``'s fixed points: the mean-field route by default until the EP route (with its double loop, which
        # passes the wiring store EP refused 59 of 65 calls on) is measured on bench-sim.
        # The quadrature at the fitted state (review F17, ``scale_mixture_ep.lattice_check``): the start lattice is
        # derived from the start's single-variant likelihoods only. Each model whose check passes is finished from the fit
        # as it stands; where any model's fitted normalizers or moments move on the refined or the extended lattice, the
        # models are fitted again on the lattice moved for them, and those are checked there.
        fit_prior = prior
        try:
            fit = fit_full_data(
                gaussian=gaussian, statistics=view, prior=fit_prior, draw_count=draw_count, working_bytes=share, seed=_seed(seed, 1),
                inference=inference, sites=fit_sites, null_scale=kappa, band=fit_band,
            )
            pending = list(range(len(columns)))
            while pending:
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
                    extended = [extended_lattice(moved, hyperparameters) for hyperparameters in moved_hyperparameters]
                    moved, moved_hyperparameters = extended[0][0], [hyperparameters for _prior, hyperparameters in extended]
                if any(checks[model].refine for model in moving):
                    halved = [halved_lattice(moved, hyperparameters) for hyperparameters in moved_hyperparameters]
                    moved, moved_hyperparameters = halved[0][0], [hyperparameters for _prior, hyperparameters in halved]
                try:
                    # Continued from the fit on the checked lattice (its hyperparameters carried to the moved lattice, its
                    # noise and its members' means), not from the prior: on bench-sim chr22 [sim] the refit from the prior
                    # repeated the whole nested search, 50 minutes of the fit, to reach the state the first fit had.
                    refit = fit_full_data(
                        gaussian=gaussian, statistics=view, prior=moved, draw_count=draw_count, working_bytes=share, seed=_seed(seed, 1),
                        inference=inference, sites=fit_sites, starts=moved_hyperparameters, start_noise=np.asarray(fit.noise_variance, dtype=np.float64),
                        start_mean=fit.member_mean, null_scale=kappa, band=fit_band,
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
) -> FittedModels:
    """Every model of ``fit_model.fit``, a group at a time (``model_groups``; see the module docstring); ``inference``
    names the fixed point (``full_data_fit.fit_full_data``): the mean-field route until the EP route (structured-genome's
    genome-scale EP) is measured on bench-sim, which then replaces it.

    Each model's completed fit is checkpointed on its own under ``checkpoint_key``; a group is fitted again unless every
    one of its models is, and a group's seed is the request's with its models' indices, so a rerun fits it the same way."""
    if any(trait_type not in (TraitType.QUANTITATIVE, TraitType.BINARY) for trait_type in trait_types):
        raise ValueError(f"unknown trait types in {list(trait_types)}")
    # Stage 0 runs a buffer on every device, and Stage 2's reads are sharded over them by LD block (``stage2_source``).
    stage2_devices = f"CUDA devices {', '.join(str(device) for device in budget.device_ids)}" if budget.device_kind == "cuda" else "the CPU"
    log(f"stage2 wiring: Stage 0 and Stage 2's reads run on {stage2_devices}")
    measurement = store_measurement(store, log_variance_offset)
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
        checkpoints = [Path(work_dir) / "checkpoints" / str(checkpoint_key) / f"model{int(model)}.npz" for model in group]
        if checkpoint_key is not None and all(checkpoint.is_file() for checkpoint in checkpoints):
            group_fits = [load_checkpoint(checkpoint, checkpoint_key) for checkpoint in checkpoints]
            log(f"stage2 wiring: resumed completed models {label}")
        else:
            # Incomplete scratch is disposable; completed models live outside this directory.
            if group_dir.exists():
                shutil.rmtree(group_dir)
            group_dir.mkdir(parents=True)
            log(f"stage2 wiring: group {index + 1} of {len(groups)}: models {label} on {rows.shape[0]:,} training rows")
            started = time.perf_counter()
            # One ledger per group's fit: its host leases and every device allocation.
            with memory_scope(budget):
                group_fits = _fit_group(
                    store,
                    np.asarray(store_columns[rows], dtype=np.int64),
                    np.asarray(covariates[rows][:, adjusted], dtype=np.float64),
                    np.asarray(targets[np.ix_(rows, group)], dtype=np.float64),
                    measurement,
                    budget,
                    group_dir,
                    _seed(seed, *group.tolist()),
                    draw_count,
                    group,
                    trait_types[first],
                    inference,
                    annotated_prior,
                )
            elapsed = time.perf_counter() - started
            if checkpoint_key is not None:
                for checkpoint, fit in zip(checkpoints, group_fits):
                    # Each model's share of its group's time, so the shares sum to the fit's.
                    save_checkpoint(checkpoint, fit, checkpoint_key, elapsed / group.shape[0])
            log(f"stage2 wiring: completed models {label} in {elapsed:.3f} seconds")
        # The group's Stage 0 LD blocks and band (its projected Grams, O(p x block cap) bytes on disk) are read by no
        # later group and not by scoring, which the fit already ran: each is removed as its group ends, so the work
        # directory never holds more than one group's.
        if group_dir.exists():
            shutil.rmtree(group_dir)
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

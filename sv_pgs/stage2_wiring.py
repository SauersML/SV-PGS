"""RUN-ONLY (run/svpgs-bench-1, fit-api): store -> fitted models, until e2e's ``full_data_fit.fit_models`` replaces it.

This is e2e's tests/test_full_data_fit wiring behind the agreed ``fit_models`` signature, run once per model:
Stage 0 on the model's own training rows and covariate columns, the start lattice from its single-variant
likelihoods, the prior with one class per variant class present and the records' log reliabilities as offsets (no
annotation groups: the design builder isn't written yet), the dual Gaussian, ``fit_full_data`` and
``scoring_models``. Models are fitted separately, so there is no cross-trait pooling of the prior's hyperparameters.
Quantitative traits only: ``fit_full_data`` has no binary likelihood yet.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.dosage_store import DosageStore
from sv_pgs.dual_solve import DualGaussian, StreamedDualSource
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.full_data_fit import FitCertificate, block_grams, covariate_residual_variance, fit_full_data, scoring_models, stage0_lattice
from sv_pgs.genotype_buffers import build_sample_layout
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import BLOCK_CAP_STEP, DosageStoreTileSource, compute_genotype_statistics, stage0_block_cap
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import MixtureHyperparameters, scale_mixture_prior
from sv_pgs.store_block_source import StoreGenotypeBlockSource


@dataclass(frozen=True)
class FittedModels:
    scoring: list[ScoringModel]
    noise_variance: F64Array
    hyperparameters: tuple[MixtureHyperparameters, ...]
    certificate: FitCertificate


def _seed(seed: int, *keys: int) -> int:
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1, dtype=np.uint64)[0])


def store_log_reliability(store: DosageStore) -> F64Array:
    """Each record's log r^2 from the store's ``quality`` column (1 where the store has none)."""
    quality = store.variant_table.annotations.get("quality")
    if quality is None:
        return np.zeros(store.n_variants)
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(quality, dtype=np.float64))


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
    grow with the cap, and a block never holds more than its chromosome's candidates."""
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
) -> tuple[ScoringModel, float, MixtureHyperparameters, FitCertificate]:
    """One quantitative model on the sorted store columns ``training_columns``, whose covariates (intercept first) and
    targets follow them."""
    # SPEC 1fca1cf: no variant is filtered by rarity or any threshold (only a derived bound may leave one out).
    config = ModelConfig(minimum_minor_allele_frequency=0.0)
    candidates = stage0_candidates(store, training_columns, log_reliability, config)
    block_cap = _block_cap(store, candidates, training_columns, covariates.shape[1], budget)
    statistics = compute_genotype_statistics(
        DosageStoreTileSource(store, candidates), training_columns, covariates, targets[:, None], config, budget, block_cap, work_dir / "ld"
    )
    kept_rows = np.asarray(statistics.active_rows, dtype=np.int64)[np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)]
    offsets = log_reliability[kept_rows]
    _classes, class_index = np.unique(store.variant_table.variant_class[kept_rows], return_inverse=True)
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
        annotation_design=np.zeros((kept_rows.shape[0], 0)),
        annotation_groups=(),
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
    fit = fit_full_data(gaussian=gaussian, statistics=statistics, prior=prior, draw_count=draw_count, working_bytes=share, seed=_seed(seed, 1))
    (scoring,) = scoring_models(fit, prior, statistics, [TraitType.QUANTITATIVE], draw_count, seed=_seed(seed, 2))
    log(f"stage2 wiring: {kept_rows.shape[0]:,} reduced columns in {statistics.ld.block_count} blocks (cap {block_cap}), {training_columns.shape[0]:,} training samples")
    return scoring, float(np.asarray(fit.noise_variance)[0]), fit.hyperparameters[0], fit.certificate


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
    log_reliability = store_log_reliability(store) if log_variance_offset is None else np.asarray(log_variance_offset, dtype=np.float64)
    scoring, noise, hyperparameters, certificates = [], [], [], []
    for model in range(training.shape[1]):
        rows = np.flatnonzero(training[:, model])
        order = np.argsort(store_columns[rows], kind="stable")
        rows = rows[order]
        adjusted = np.asarray(covariate_columns[model], dtype=bool)
        model_dir = Path(work_dir) / f"model{model}"
        model_dir.mkdir()
        fitted, model_noise, model_hyperparameters, certificate = _fit_one(
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
        alpha[adjusted] = fitted.alpha
        scoring.append(dataclasses.replace(fitted, alpha=alpha))
        noise.append(model_noise)
        hyperparameters.append(model_hyperparameters)
        certificates.append(certificate)
    return FittedModels(
        scoring=scoring, noise_variance=np.array(noise), hyperparameters=tuple(hyperparameters), certificate=_merged_certificate(certificates)
    )

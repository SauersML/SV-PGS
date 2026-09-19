"""Fast architecture: one genotype pass, then the LD-space EP-EB fit, for every trait at once.

Stage 0 (phenotype independent apart from the target columns riding the same pass) reads the
dosage store once and writes the covariate-projected LD blocks (``genotype_statistics``).
Stage 1 fits every trait's empirical-Bayes hyperparameters and posterior in LD space
(``ld_space_fit``, expectation propagation). The result is expressed as one
``fast_scoring.ScoringModel`` per trait, in store rows and signed-code units, so one read of
the store scores every trait.

Stage 1 drops the LD between blocks, so its fit is a warm start: SPEC accepts a fit only after
Stage 2 certifies it on the individual-level data. ``FastFitResult.certified`` says which.

Binary traits enter Stage 1 through the working model at the covariate-only fit: the target
column that rides Stage 0 is y - pi, with pi from the covariate logistic regression, whose
score equations make C'(y - pi) = 0, so its projected score is exactly X~'(y - pi).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.fast_scoring import ScoringModel
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics, GenotypeTileSource, compute_genotype_statistics
from sv_pgs.ld_space_fit import (
    LDPriorHypermodel,
    LDSpaceFit,
    TraitStatistics,
    binary_statistics_at,
    fit_ld_space,
    quantitative_trait_statistics,
)
from sv_pgs.progress import log

# The local-scale spike shape is pooled and fixed (theory-inference E1b, E7): a0 = 1/2.
_SHAPE_A = 0.5
# Tail shapes start at the Gibbs-matched SNV value; log b_c ~ N(log b_bar, 0.25^2) (lead ruling 00:48Z).
_INITIAL_SHAPE_B = 0.5
_SHAPE_B_POOLING_VARIANCE = 0.25**2
# Class offsets carry an N(0, 1) hyperprior against weak-class plateau drift (lead ruling 00:48Z).
_CLASS_OFFSET_PRIOR_PRECISION = 1.0
_LOGISTIC_TOLERANCE = 1e-10
_LOGISTIC_MAXIMUM_ITERATIONS = 100


@dataclass(frozen=True, slots=True)
class CovariateLogisticFit:
    """The covariate-only logistic regression of a binary trait (intercept inside ``covariates``)."""

    coefficients: NDArray[np.float64]
    fitted_probability: NDArray[np.float64]


@dataclass(slots=True)
class FastFitResult:
    """Every trait's Stage 1 fit and its scoring model; ``certified`` is False until Stage 2 runs."""

    statistics: GenotypeSufficientStatistics
    hypermodel: LDPriorHypermodel
    fits: list[LDSpaceFit]
    models: list[ScoringModel]
    certified: bool
    stage_seconds: dict[str, float] = field(default_factory=dict)


def covariate_logistic_fit(covariates: NDArray[np.float64], labels: NDArray[np.float64]) -> CovariateLogisticFit:
    """Newton's method with step halving on the (strictly concave) covariate log-likelihood."""
    design = np.asarray(covariates, dtype=np.float64)
    outcome = np.asarray(labels, dtype=np.float64)
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("binary targets must be coded 0/1")
    coefficients = np.zeros(design.shape[1], dtype=np.float64)

    def log_likelihood(candidate: NDArray[np.float64]) -> float:
        linear = design @ candidate
        return float(outcome @ linear - np.sum(np.logaddexp(0.0, linear)))

    current = log_likelihood(coefficients)
    for _ in range(_LOGISTIC_MAXIMUM_ITERATIONS):
        probability = 1.0 / (1.0 + np.exp(-(design @ coefficients)))
        gradient = design.T @ (outcome - probability)
        hessian = (design * (probability * (1.0 - probability))[:, None]).T @ design
        step = np.linalg.solve(hessian, gradient)
        length = 1.0
        while True:
            candidate = coefficients + length * step
            value = log_likelihood(candidate)
            if value >= current or length < 1e-12:
                break
            length *= 0.5
        coefficients, previous, current = candidate, current, value
        if abs(current - previous) <= _LOGISTIC_TOLERANCE * (1.0 + abs(current)):
            break
    probability = 1.0 / (1.0 + np.exp(-(design @ coefficients)))
    return CovariateLogisticFit(coefficients=coefficients, fitted_probability=probability)


def class_hypermodel(reduced_class_index: NDArray[np.int64], class_names: Sequence[str]) -> LDPriorHypermodel:
    """The prior's class structure: one offset per class present beyond the first, N(0, 1) each.

    Later SV-context, reliability and external-annotation columns join this design
    (idea-svprior owns its content).
    """
    class_index = np.asarray(reduced_class_index, dtype=np.int64)
    present = np.unique(class_index)
    offset_classes = present[1:]
    design = (class_index[:, None] == offset_classes[None, :]).astype(np.float64)
    return LDPriorHypermodel(
        annotation_design=design,
        annotation_prior_mean=np.zeros(offset_classes.shape[0], dtype=np.float64),
        annotation_prior_precision=np.full(offset_classes.shape[0], _CLASS_OFFSET_PRIOR_PRECISION, dtype=np.float64),
        log_variance_offset=np.zeros(class_index.shape[0], dtype=np.float64),
        variant_class_index=class_index,
        class_names=tuple(class_names),
        shape_a=_SHAPE_A,
        initial_shape_b=np.full(len(class_names), _INITIAL_SHAPE_B, dtype=np.float64),
        shape_b_pooling_variance=_SHAPE_B_POOLING_VARIANCE,
    )


def _reduced_store_rows(statistics: GenotypeSufficientStatistics) -> NDArray[np.int64]:
    """The store row of each reduced column's representative."""
    return statistics.active_rows[np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)]


def _expanded_coefficients(statistics: GenotypeSufficientStatistics, reduced_beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each tie group's effect split equally over its members (equal prior variance within a group)."""
    tie_map = statistics.tie_map
    group_weights = [
        np.full(group.member_indices.shape[0], 1.0 / group.member_indices.shape[0], dtype=np.float32)
        for group in tie_map.reduced_to_group
    ]
    return np.asarray(tie_map.expand_coefficients(np.asarray(reduced_beta, dtype=np.float64), group_weights), dtype=np.float64)


def _quantitative_alpha(statistics: GenotypeSufficientStatistics, trait_index: int, reduced_beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Covariate coefficients given the genetic effects: (C'C)^-1 (C'y - (X'C)' beta)."""
    ld = statistics.ld
    genetic_cross = np.zeros(statistics.covariate_gram.shape[0], dtype=np.float64)
    boundaries = np.asarray(ld.block_boundaries, dtype=np.int64)
    for block_index in range(ld.block_count):
        block_beta = reduced_beta[boundaries[block_index] : boundaries[block_index + 1]]
        genetic_cross += ld.block(block_index).covariate_cross.T @ block_beta
    return np.linalg.solve(statistics.covariate_gram, statistics.covariate_target[:, trait_index] - genetic_cross)


def fit_fast(
    *,
    source: GenotypeTileSource,
    sample_indices: NDArray[np.int64],
    covariates: NDArray[np.float64],
    targets: NDArray[np.float64],
    trait_types: Sequence[TraitType],
    class_index_of_store_row: NDArray[np.int64],
    class_names: Sequence[str],
    config: ModelConfig,
    budget: ComputeBudget,
    block_cap: int,
    work_dir: Path,
) -> FastFitResult:
    """Stage 0 and Stage 1 for every trait on the training samples ``sample_indices``.

    ``covariates`` ``[n, k]`` include the intercept column; ``targets`` ``[n, T]`` hold the
    phenotypes (0/1 for binary traits). Rows follow ``sample_indices``.
    """
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    target_matrix = np.asarray(targets, dtype=np.float64)
    if target_matrix.shape[1] != len(trait_types):
        raise ValueError("targets need one column per trait type")
    if not np.allclose(covariate_matrix[:, 0], 1.0):
        raise ValueError("the first covariate column must be the intercept")
    stage_seconds: dict[str, float] = {}
    started = time.monotonic()
    logistic_fits: dict[int, CovariateLogisticFit] = {}
    pass_targets = target_matrix.copy()
    for trait_index, trait_type in enumerate(trait_types):
        if trait_type == TraitType.BINARY:
            logistic_fits[trait_index] = covariate_logistic_fit(covariate_matrix, target_matrix[:, trait_index])
            pass_targets[:, trait_index] = target_matrix[:, trait_index] - logistic_fits[trait_index].fitted_probability
    stage_seconds["covariate_fits"] = time.monotonic() - started

    started = time.monotonic()
    work_dir.mkdir(parents=True, exist_ok=True)
    statistics = compute_genotype_statistics(
        source, np.asarray(sample_indices, dtype=np.int64), covariate_matrix, pass_targets, config, budget, block_cap, work_dir,
    )
    stage_seconds["stage0"] = time.monotonic() - started

    reduced_rows = _reduced_store_rows(statistics)
    hypermodel = class_hypermodel(np.asarray(class_index_of_store_row, dtype=np.int64)[reduced_rows], class_names)
    covariate_count = int(covariate_matrix.shape[1])
    trait_statistics: list[TraitStatistics] = []
    for trait_index, trait_type in enumerate(trait_types):
        if trait_type == TraitType.BINARY:
            quantitative_view = quantitative_trait_statistics(statistics, trait_index)
            trait_statistics.append(binary_statistics_at(
                residual_score=quantitative_view.score,
                gram_times_expansion=np.zeros_like(quantitative_view.score),
                fitted_probability=logistic_fits[trait_index].fitted_probability,
                covariate_count=covariate_count,
            ))
        else:
            trait_statistics.append(quantitative_trait_statistics(statistics, trait_index))

    started = time.monotonic()
    fits = fit_ld_space(statistics.ld, trait_statistics, hypermodel, budget)
    stage_seconds["stage1"] = time.monotonic() - started
    for trait_index, fit in enumerate(fits):
        log(f"fast fit trait {trait_index}: stage 1 {fit.scheme} converged={fit.converged} after {fit.passes} passes")

    models = []
    for trait_index, (trait_type, fit) in enumerate(zip(trait_types, fits, strict=True)):
        reduced_beta = np.asarray(fit.posterior_mean, dtype=np.float64)
        covariate_coefficients = (
            logistic_fits[trait_index].coefficients
            if trait_type == TraitType.BINARY
            else _quantitative_alpha(statistics, trait_index, reduced_beta)
        )
        models.append(ScoringModel(
            store_rows=np.asarray(statistics.active_rows, dtype=np.int64),
            signed_means=np.asarray(statistics.means, dtype=np.float64),
            signed_scales=np.asarray(statistics.scales, dtype=np.float64),
            coefficients=_expanded_coefficients(statistics, reduced_beta),
            alpha=np.asarray(covariate_coefficients, dtype=np.float64),
            trait_type=trait_type,
            predictive_intercept_shift=0.0,
        ))
    return FastFitResult(
        statistics=statistics, hypermodel=hypermodel, fits=fits, models=models, certified=False, stage_seconds=stage_seconds,
    )

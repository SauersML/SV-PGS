from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import VariantRecord
from sv_pgs.genotype import StandardizedGenotypeMatrix
from sv_pgs.mixture_inference import (
    _apply_binary_intercept_calibration,
    _fit_collapsed_posterior,
    _initialize_alpha_state,
)
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.progress import log, mem

STAGE1_NULL_MODEL_MIN_SAMPLE_COUNT = 1_024
STAGE1_NULL_MODEL_MIN_VARIANT_COUNT = 2_048
STAGE1_SMALL_PROBLEM_VALIDATION_FRACTION = 0.2
STAGE1_SMALL_PROBLEM_MIN_VALIDATION_COUNT = 8
STAGE1_SMALL_PROBLEM_MIN_TRAIN_COUNT = 24
STAGE1_SMALL_PROBLEM_MIN_RELATIVE_GAIN = 0.01


@dataclass(slots=True)
class NullModelFit:
    alpha: np.ndarray
    full_coefficients: np.ndarray
    selected_variant_indices: np.ndarray
    linear_predictor: np.ndarray


def _minor_allele_frequency(allele_frequency: float) -> float:
    normalized_frequency = float(np.clip(allele_frequency, 0.0, 1.0))
    return min(normalized_frequency, 1.0 - normalized_frequency)


def _select_evenly_spaced_indices(indices: np.ndarray, selection_count: int) -> np.ndarray:
    resolved_count = min(max(int(selection_count), 0), int(indices.shape[0]))
    if resolved_count <= 0:
        return np.zeros(0, dtype=np.int32)
    if resolved_count >= indices.shape[0]:
        return np.asarray(indices, dtype=np.int32)
    selection_positions = np.linspace(
        0,
        indices.shape[0] - 1,
        resolved_count,
        dtype=np.int32,
    )
    return np.asarray(indices[selection_positions], dtype=np.int32)


def _balanced_stage1_variant_subsample(
    eligible_indices: np.ndarray,
    variant_records: Sequence[VariantRecord],
    max_variants: int,
) -> np.ndarray:
    resolved_max_variants = min(max(int(max_variants), 0), int(eligible_indices.shape[0]))
    if resolved_max_variants <= 0:
        return np.zeros(0, dtype=np.int32)
    if resolved_max_variants >= eligible_indices.shape[0]:
        return np.asarray(eligible_indices, dtype=np.int32)
    chromosome_to_indices: dict[str, list[int]] = {}
    for variant_index in np.asarray(eligible_indices, dtype=np.int32):
        chromosome_to_indices.setdefault(str(variant_records[int(variant_index)].chromosome), []).append(int(variant_index))
    chromosome_entries = sorted(
        chromosome_to_indices.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    total_eligible = int(eligible_indices.shape[0])
    quotas = {
        chromosome: int(np.floor(len(chromosome_indices) * resolved_max_variants / total_eligible))
        for chromosome, chromosome_indices in chromosome_entries
    }
    assigned = sum(quotas.values())
    if assigned < resolved_max_variants:
        fractional_order = sorted(
            chromosome_entries,
            key=lambda item: (
                -(
                    len(item[1]) * resolved_max_variants / total_eligible
                    - np.floor(len(item[1]) * resolved_max_variants / total_eligible)
                ),
                -len(item[1]),
                item[0],
            ),
        )
        for chromosome, chromosome_indices in fractional_order:
            if assigned >= resolved_max_variants:
                break
            if quotas[chromosome] >= len(chromosome_indices):
                continue
            quotas[chromosome] += 1
            assigned += 1
    selected_blocks = [
        _select_evenly_spaced_indices(np.asarray(chromosome_indices, dtype=np.int32), quotas[chromosome])
        for chromosome, chromosome_indices in chromosome_entries
        if quotas[chromosome] > 0
    ]
    if not selected_blocks:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(np.sort(np.concatenate(selected_blocks)), dtype=np.int32)


def _fit_stage1_covariates_only(
    covariate_matrix: np.ndarray,
    target_array: np.ndarray,
    config: ModelConfig,
    *,
    variant_count: int,
    selected_variant_indices: np.ndarray,
) -> NullModelFit:
    alpha = _initialize_alpha_state(
        covariate_matrix=covariate_matrix,
        targets=target_array,
        trait_type=config.trait_type,
    )
    linear_predictor = np.asarray(covariate_matrix @ alpha, dtype=np.float64)
    if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
        linear_predictor_state = _apply_binary_intercept_calibration(
            posterior_state=type(
                "_Stage1PosteriorState",
                (),
                {
                    "alpha": alpha,
                    "beta": np.zeros(0, dtype=np.float64),
                    "beta_variance": np.zeros(0, dtype=np.float64),
                    "linear_predictor": linear_predictor,
                    "collapsed_objective": float(
                        np.mean(
                            target_array * np.log(np.asarray(stable_sigmoid(linear_predictor), dtype=np.float64) + 1e-12)
                            + (1.0 - target_array) * np.log(1.0 - np.asarray(stable_sigmoid(linear_predictor), dtype=np.float64) + 1e-12)
                        )
                    ),
                    "sigma_error2": 1.0,
                },
            )(),
            targets=target_array,
        )
        alpha = np.asarray(linear_predictor_state.alpha, dtype=np.float64)
        linear_predictor = np.asarray(linear_predictor_state.linear_predictor, dtype=np.float64)
    return NullModelFit(
        alpha=np.asarray(alpha, dtype=np.float32),
        full_coefficients=np.zeros(variant_count, dtype=np.float32),
        selected_variant_indices=np.asarray(selected_variant_indices, dtype=np.int32),
        linear_predictor=np.asarray(linear_predictor, dtype=np.float32),
    )


def _stage1_validation_loss(
    targets: np.ndarray,
    linear_predictor: np.ndarray,
    trait_type: TraitType,
) -> float:
    target_array = np.asarray(targets, dtype=np.float64)
    predictor = np.asarray(linear_predictor, dtype=np.float64)
    if trait_type == TraitType.BINARY:
        probabilities = np.asarray(stable_sigmoid(predictor), dtype=np.float64)
        return float(
            -np.mean(
                target_array * np.log(probabilities + 1e-12)
                + (1.0 - target_array) * np.log(1.0 - probabilities + 1e-12)
            )
        )
    residual = target_array - predictor
    return float(np.mean(residual * residual))


def _stage1_small_problem_accepts_genotype_offset(
    standardized_stage1_matrix: np.ndarray,
    covariate_matrix: np.ndarray,
    target_array: np.ndarray,
    config: ModelConfig,
) -> bool:
    sample_count = int(target_array.shape[0])
    validation_count = max(
        STAGE1_SMALL_PROBLEM_MIN_VALIDATION_COUNT,
        int(np.ceil(sample_count * STAGE1_SMALL_PROBLEM_VALIDATION_FRACTION)),
    )
    validation_count = min(validation_count, sample_count - 1)
    train_count = sample_count - validation_count
    if train_count < STAGE1_SMALL_PROBLEM_MIN_TRAIN_COUNT:
        return False
    random_generator = np.random.default_rng(config.random_seed)
    permutation = random_generator.permutation(sample_count)
    validation_indices = np.sort(permutation[:validation_count])
    training_indices = np.sort(permutation[validation_count:])
    train_covariates = np.asarray(covariate_matrix[training_indices], dtype=np.float64)
    validation_covariates = np.asarray(covariate_matrix[validation_indices], dtype=np.float64)
    train_targets = np.asarray(target_array[training_indices], dtype=np.float64)
    validation_targets = np.asarray(target_array[validation_indices], dtype=np.float64)
    train_genotypes = np.asarray(standardized_stage1_matrix[training_indices], dtype=np.float32)
    validation_genotypes = np.asarray(standardized_stage1_matrix[validation_indices], dtype=np.float32)

    baseline_alpha = _initialize_alpha_state(train_covariates, train_targets, config.trait_type)
    baseline_validation_predictor = np.asarray(validation_covariates @ baseline_alpha, dtype=np.float64)
    baseline_loss = _stage1_validation_loss(
        targets=validation_targets,
        linear_predictor=baseline_validation_predictor,
        trait_type=config.trait_type,
    )
    stage1_state = _fit_collapsed_posterior(
        genotype_matrix=train_genotypes,
        covariate_matrix=train_covariates,
        targets=train_targets,
        reduced_prior_variances=np.full(train_genotypes.shape[1], config.stage1_prior_variance, dtype=np.float64),
        sigma_error2=1.0,
        alpha_init=_initialize_alpha_state(train_covariates, train_targets, config.trait_type),
        beta_init=np.zeros(train_genotypes.shape[1], dtype=np.float64),
        trait_type=config.trait_type,
        config=config,
        compute_logdet=False,
        compute_beta_variance=False,
    )
    if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
        stage1_state = _apply_binary_intercept_calibration(
            posterior_state=stage1_state,
            targets=train_targets,
        )
    stage1_validation_predictor = np.asarray(
        validation_covariates @ stage1_state.alpha + validation_genotypes @ stage1_state.beta,
        dtype=np.float64,
    )
    stage1_loss = _stage1_validation_loss(
        targets=validation_targets,
        linear_predictor=stage1_validation_predictor,
        trait_type=config.trait_type,
    )
    return stage1_loss < baseline_loss * (1.0 - STAGE1_SMALL_PROBLEM_MIN_RELATIVE_GAIN)


def select_stage1_variant_indices(
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> np.ndarray:
    if not config.enable_stage1_null_model:
        return np.zeros(0, dtype=np.int32)
    eligible_indices = np.asarray(
        [
            variant_index
            for variant_index, record in enumerate(variant_records)
            if _minor_allele_frequency(record.allele_frequency) >= config.stage1_min_minor_allele_frequency
        ],
        dtype=np.int32,
    )
    if eligible_indices.shape[0] <= int(config.stage1_max_variants):
        return eligible_indices
    if int(config.stage1_max_variants) == 0:
        return np.zeros(0, dtype=np.int32)
    return _balanced_stage1_variant_subsample(
        eligible_indices=eligible_indices,
        variant_records=variant_records,
        max_variants=int(config.stage1_max_variants),
    )


def fit_stage1_null_model(
    standardized_genotypes: StandardizedGenotypeMatrix,
    covariates: np.ndarray,
    targets: np.ndarray,
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> NullModelFit:
    target_array = np.asarray(targets, dtype=np.float64)
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    selected_variant_indices = select_stage1_variant_indices(variant_records, config)
    small_problem = (
        target_array.shape[0] < STAGE1_NULL_MODEL_MIN_SAMPLE_COUNT
        or len(variant_records) < STAGE1_NULL_MODEL_MIN_VARIANT_COUNT
    )
    if not config.enable_stage1_null_model:
        stage1_reason = "disabled"
    elif small_problem:
        stage1_reason = f"small_problem(n={target_array.shape[0]},p={len(variant_records)})"
    else:
        stage1_reason = f"min_maf={config.stage1_min_minor_allele_frequency:.4f}"
    log(
        "stage-1 null model: "
        + f"selected {selected_variant_indices.shape[0]}/{len(variant_records)} common variants "
        + f"({stage1_reason})  mem={mem()}"
    )
    if selected_variant_indices.shape[0] == 0:
        return _fit_stage1_covariates_only(
            covariate_matrix=covariate_matrix,
            target_array=target_array,
            config=config,
            variant_count=len(variant_records),
            selected_variant_indices=selected_variant_indices,
        )
    stage1_genotypes = standardized_genotypes.subset(selected_variant_indices)
    if small_problem:
        stage1_dense_matrix = np.asarray(stage1_genotypes.materialize(), dtype=np.float32)
        if not _stage1_small_problem_accepts_genotype_offset(
            standardized_stage1_matrix=stage1_dense_matrix,
            covariate_matrix=covariate_matrix,
            target_array=target_array,
            config=config,
        ):
            log("stage-1 null model: dropping genotype offset after internal validation gate")
            return _fit_stage1_covariates_only(
                covariate_matrix=covariate_matrix,
                target_array=target_array,
                config=config,
                variant_count=len(variant_records),
                selected_variant_indices=np.zeros(0, dtype=np.int32),
            )
    stage1_state = _fit_collapsed_posterior(
        genotype_matrix=stage1_genotypes,
        covariate_matrix=covariate_matrix,
        targets=target_array,
        reduced_prior_variances=np.full(selected_variant_indices.shape[0], config.stage1_prior_variance, dtype=np.float64),
        sigma_error2=1.0,
        alpha_init=_initialize_alpha_state(covariate_matrix, target_array, config.trait_type),
        beta_init=np.zeros(selected_variant_indices.shape[0], dtype=np.float64),
        trait_type=config.trait_type,
        config=config,
        compute_logdet=False,
        compute_beta_variance=False,
    )
    if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
        stage1_state = _apply_binary_intercept_calibration(
            posterior_state=stage1_state,
            targets=target_array,
        )
    full_coefficients = np.zeros(len(variant_records), dtype=np.float32)
    full_coefficients[selected_variant_indices] = np.asarray(stage1_state.beta, dtype=np.float32)
    return NullModelFit(
        alpha=np.asarray(stage1_state.alpha, dtype=np.float32),
        full_coefficients=full_coefficients,
        selected_variant_indices=selected_variant_indices,
        linear_predictor=np.asarray(stage1_state.linear_predictor, dtype=np.float32),
    )

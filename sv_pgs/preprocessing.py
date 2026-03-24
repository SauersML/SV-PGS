from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord


@dataclass(slots=True)
class Preprocessor:
    means: np.ndarray
    scales: np.ndarray

    def transform(self, genotypes: np.ndarray) -> np.ndarray:
        genotype_matrix = np.asarray(genotypes, dtype=np.float32)
        genotype_matrix = _impute_missing_values(genotype_matrix, self.means)
        return (genotype_matrix - self.means) / self.scales


def fit_preprocessor(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
) -> PreparedArrays:
    genotype_matrix = np.asarray(genotypes, dtype=np.float32)
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32).reshape(-1)
    if genotype_matrix.ndim != 2:
        raise ValueError("genotypes must be 2D.")
    if covariate_matrix.ndim != 2:
        raise ValueError("covariates must be 2D.")
    if (
        genotype_matrix.shape[0] != covariate_matrix.shape[0]
        or genotype_matrix.shape[0] != target_array.shape[0]
    ):
        raise ValueError("genotypes, covariates, and targets must share sample dimension.")

    means = np.nanmean(genotype_matrix, axis=0, dtype=np.float64).astype(np.float32)
    means = np.where(np.isnan(means), 0.0, means)

    filled_genotypes = _impute_missing_values(genotype_matrix, means)

    centered_genotypes = filled_genotypes - means
    scales = np.sqrt(np.mean(centered_genotypes * centered_genotypes, axis=0, dtype=np.float64)).astype(np.float32)
    scales = np.where(scales < config.min_scale, 1.0, scales)
    standardized_genotypes = centered_genotypes / scales

    return PreparedArrays(
        genotypes=standardized_genotypes,
        covariates=covariate_matrix,
        targets=target_array,
        means=means,
        scales=scales,
    )


def _impute_missing_values(genotype_matrix: np.ndarray, means: np.ndarray) -> np.ndarray:
    missing_mask = np.isnan(genotype_matrix)
    if not missing_mask.any():
        return genotype_matrix
    imputed = genotype_matrix.copy()
    imputed[missing_mask] = np.take(means, np.where(missing_mask)[1])
    return imputed


def build_tie_map(genotypes: np.ndarray, records: Sequence[VariantRecord]) -> TieMap:
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be 2D.")
    if genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")

    exact_signature_to_group: dict[tuple[VariantClass, tuple[int, ...]], int] = {}
    sign_flipped_signature_to_group: dict[tuple[VariantClass, tuple[int, ...]], int] = {}
    tie_groups: list[TieGroup] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(genotypes.shape[1], -1, dtype=np.int32)

    rounded_genotypes = np.rint(genotypes * 1_000_000.0).astype(np.int64, copy=False)
    for variant_index, variant_record in enumerate(records):
        signature = tuple(rounded_genotypes[:, variant_index].tolist())
        exact_key = (variant_record.variant_class, signature)
        sign_flipped_key = (
            variant_record.variant_class,
            tuple((-rounded_genotypes[:, variant_index]).tolist()),
        )

        if exact_key in exact_signature_to_group:
            reduced_index = exact_signature_to_group[exact_key]
            tie_group = tie_groups[reduced_index]
            tie_group.member_indices = np.append(tie_group.member_indices, variant_index)
            tie_group.signs = np.append(tie_group.signs, 1.0)
            original_to_reduced[variant_index] = reduced_index
            continue

        if exact_key in sign_flipped_signature_to_group:
            reduced_index = sign_flipped_signature_to_group[exact_key]
            tie_group = tie_groups[reduced_index]
            tie_group.member_indices = np.append(tie_group.member_indices, variant_index)
            tie_group.signs = np.append(tie_group.signs, -1.0)
            original_to_reduced[variant_index] = reduced_index
            continue

        reduced_index = len(tie_groups)
        tie_groups.append(
            TieGroup(
                representative_index=variant_index,
                member_indices=np.asarray([variant_index], dtype=np.int32),
                signs=np.asarray([1.0], dtype=np.float32),
            )
        )
        kept_variant_indices.append(variant_index)
        exact_signature_to_group[exact_key] = reduced_index
        sign_flipped_signature_to_group[sign_flipped_key] = reduced_index
        original_to_reduced[variant_index] = reduced_index

    return TieMap(
        kept_indices=np.asarray(kept_variant_indices, dtype=np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=tie_groups,
    )

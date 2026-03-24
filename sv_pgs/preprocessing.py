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
        imputed_genotypes = _impute_missing_values(genotype_matrix, self.means)
        return (imputed_genotypes - self.means) / self.scales


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
    if genotype_matrix.shape[0] != covariate_matrix.shape[0] or genotype_matrix.shape[0] != target_array.shape[0]:
        raise ValueError("genotypes, covariates, and targets must share sample dimension.")

    means = np.nanmean(genotype_matrix, axis=0, dtype=np.float64).astype(np.float32)
    means = np.where(np.isnan(means), 0.0, means)
    imputed_genotypes = _impute_missing_values(genotype_matrix, means)
    centered_genotypes = imputed_genotypes - means
    scales = np.sqrt(np.mean(centered_genotypes * centered_genotypes, axis=0, dtype=np.float64)).astype(np.float32)
    scales = np.where(scales < config.minimum_scale, 1.0, scales)

    return PreparedArrays(
        genotypes=centered_genotypes / scales,
        covariates=covariate_matrix,
        targets=target_array,
        means=means,
        scales=scales,
    )


def _impute_missing_values(genotype_matrix: np.ndarray, means: np.ndarray) -> np.ndarray:
    missing_mask = np.isnan(genotype_matrix)
    if not missing_mask.any():
        return genotype_matrix
    imputed_genotypes = genotype_matrix.copy()
    imputed_genotypes[missing_mask] = np.take(means, np.where(missing_mask)[1])
    return imputed_genotypes


def select_active_variant_indices(
    genotype_matrix: np.ndarray,
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> np.ndarray:
    active_flags = np.ones(len(variant_records), dtype=bool)
    structural_variant_classes = set(config.structural_variant_classes())
    for variant_index, variant_record in enumerate(variant_records):
        if variant_record.variant_class not in structural_variant_classes:
            continue
        carrier_count = _carrier_count(genotype_matrix[:, variant_index])
        if carrier_count < config.minimum_structural_variant_carriers:
            active_flags[variant_index] = False
    return np.where(active_flags)[0].astype(np.int32)


def _carrier_count(variant_values: np.ndarray) -> int:
    non_missing_values = variant_values[~np.isnan(variant_values)]
    return int(np.count_nonzero(np.abs(non_missing_values) > 0.0))


def build_tie_map(
    genotypes: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> TieMap:
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be 2D.")
    if genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")

    exact_signature_to_group: dict[tuple[int, ...], int] = {}
    sign_flipped_signature_to_group: dict[tuple[int, ...], int] = {}
    tie_groups: list[TieGroup] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(genotypes.shape[1], -1, dtype=np.int32)

    signature_scale = 10**config.duplicate_signature_decimals
    rounded_genotypes = np.rint(genotypes * signature_scale).astype(np.int64, copy=False)
    for variant_index in range(genotypes.shape[1]):
        genotype_signature = tuple(rounded_genotypes[:, variant_index].tolist())
        sign_flipped_signature = tuple((-rounded_genotypes[:, variant_index]).tolist())

        if genotype_signature in exact_signature_to_group:
            reduced_index = exact_signature_to_group[genotype_signature]
            tie_group = tie_groups[reduced_index]
            tie_group.member_indices = np.append(tie_group.member_indices, variant_index)
            tie_group.signs = np.append(tie_group.signs, 1.0)
            original_to_reduced[variant_index] = reduced_index
            continue

        if genotype_signature in sign_flipped_signature_to_group:
            reduced_index = sign_flipped_signature_to_group[genotype_signature]
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
        exact_signature_to_group[genotype_signature] = reduced_index
        sign_flipped_signature_to_group[sign_flipped_signature] = reduced_index
        original_to_reduced[variant_index] = reduced_index

    return TieMap(
        kept_indices=np.asarray(kept_variant_indices, dtype=np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=tie_groups,
    )


def collapse_tie_groups(
    records: Sequence[VariantRecord],
    tie_map: TieMap,
) -> list[VariantRecord]:
    collapsed_records: list[VariantRecord] = []
    for tie_group in tie_map.reduced_to_group:
        member_records = [records[int(member_index)] for member_index in tie_group.member_indices]
        unique_variant_classes, class_membership = _class_membership(member_records)
        latent_variant_class = (
            unique_variant_classes[0] if len(unique_variant_classes) == 1 else VariantClass.OTHER_COMPLEX_SV
        )
        collapsed_records.append(
            VariantRecord(
                variant_id=member_records[0].variant_id,
                variant_class=latent_variant_class,
                chromosome=member_records[0].chromosome,
                position=int(np.min([member_record.position for member_record in member_records])),
                length=float(np.mean([member_record.length for member_record in member_records])),
                allele_frequency=float(np.mean([member_record.allele_frequency for member_record in member_records])),
                quality=float(np.mean([member_record.quality for member_record in member_records])),
                is_repeat=any(member_record.is_repeat for member_record in member_records),
                is_copy_number=any(member_record.is_copy_number for member_record in member_records),
                prior_class_members=tuple(unique_variant_classes),
                prior_class_membership=tuple(class_membership.tolist()),
            )
        )
    return collapsed_records


def _class_membership(member_records: Sequence[VariantRecord]) -> tuple[list[VariantClass], np.ndarray]:
    class_counts: dict[VariantClass, int] = {}
    for member_record in member_records:
        class_counts[member_record.variant_class] = class_counts.get(member_record.variant_class, 0) + 1
    unique_variant_classes = sorted(class_counts, key=lambda variant_class: variant_class.value)
    class_weights = np.asarray(
        [class_counts[variant_class] for variant_class in unique_variant_classes],
        dtype=np.float64,
    )
    class_weights /= np.sum(class_weights)
    return unique_variant_classes, class_weights

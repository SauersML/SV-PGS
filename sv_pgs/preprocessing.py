from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord
from sv_pgs.genotype import (
    DenseRawGenotypeMatrix,
    RawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    as_raw_genotype_matrix,
)


@dataclass(slots=True)
class Preprocessor:
    means: np.ndarray
    scales: np.ndarray

    def transform(self, genotypes: RawGenotypeMatrix | np.ndarray) -> StandardizedGenotypeMatrix | np.ndarray:
        raw_genotypes = as_raw_genotype_matrix(genotypes)
        standardized = raw_genotypes.standardized(self.means, self.scales)
        if isinstance(genotypes, np.ndarray):
            return standardized.materialize()
        return standardized


def fit_preprocessor(
    genotypes: RawGenotypeMatrix | np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
) -> PreparedArrays:
    raw_genotypes = as_raw_genotype_matrix(genotypes)
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32).reshape(-1)
    if covariate_matrix.ndim != 2:
        raise ValueError("covariates must be 2D.")
    if raw_genotypes.shape[0] != covariate_matrix.shape[0] or raw_genotypes.shape[0] != target_array.shape[0]:
        raise ValueError("genotypes, covariates, and targets must share sample dimension.")

    variant_count = raw_genotypes.shape[1]
    sums = np.zeros(variant_count, dtype=np.float64)
    non_missing_counts = np.zeros(variant_count, dtype=np.int64)
    for batch in raw_genotypes.iter_column_batches(batch_size=config.genotype_batch_size):
        mask = ~np.isnan(batch.values)
        observed = np.where(mask, batch.values, 0.0)
        sums[batch.variant_indices] = np.sum(observed, axis=0, dtype=np.float64)
        non_missing_counts[batch.variant_indices] = np.sum(mask, axis=0, dtype=np.int64)

    means = np.divide(
        sums,
        np.maximum(non_missing_counts, 1),
        out=np.zeros_like(sums),
        where=non_missing_counts > 0,
    )

    centered_sum_squares = np.zeros(variant_count, dtype=np.float64)
    for batch in raw_genotypes.iter_column_batches(batch_size=config.genotype_batch_size):
        batch_means = means[batch.variant_indices]
        imputed = np.where(np.isnan(batch.values), batch_means[None, :], batch.values)
        centered = imputed - batch_means[None, :]
        centered_sum_squares[batch.variant_indices] = np.sum(centered * centered, axis=0, dtype=np.float64)

    scales = np.sqrt(centered_sum_squares / max(raw_genotypes.shape[0], 1))
    scales = np.where(scales < config.minimum_scale, 1.0, scales)

    return PreparedArrays(
        covariates=np.asarray(covariate_matrix, dtype=np.float32),
        targets=np.asarray(target_array, dtype=np.float32),
        means=np.asarray(means, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
    )


def select_active_variant_indices(
    genotype_matrix: RawGenotypeMatrix | np.ndarray,
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> np.ndarray:
    raw_genotypes = as_raw_genotype_matrix(genotype_matrix)
    active_flags = np.ones(len(variant_records), dtype=bool)
    structural_variant_classes = set(config.structural_variant_classes())
    for batch in raw_genotypes.iter_column_batches(batch_size=config.genotype_batch_size):
        for local_index, variant_index in enumerate(batch.variant_indices):
            variant_record = variant_records[int(variant_index)]
            if variant_record.variant_class not in structural_variant_classes:
                continue
            support_count = _structural_variant_support(
                raw_variant_values=batch.values[:, local_index],
                variant_record=variant_record,
            )
            if support_count < config.minimum_structural_variant_carriers:
                active_flags[int(variant_index)] = False
    return np.where(active_flags)[0].astype(np.int32)


def _structural_variant_support(
    raw_variant_values: np.ndarray,
    variant_record: VariantRecord,
) -> int:
    if variant_record.training_support is not None:
        return int(variant_record.training_support)
    return _infer_support_count_from_raw_genotypes(raw_variant_values, variant_record)


def _infer_support_count_from_raw_genotypes(
    raw_variant_values: np.ndarray,
    variant_record: VariantRecord,
) -> int:
    non_missing_values = np.asarray(raw_variant_values[~np.isnan(raw_variant_values)], dtype=np.float64)
    if non_missing_values.size == 0:
        return 0
    rounded_values = np.rint(non_missing_values)
    if not np.allclose(non_missing_values, rounded_values, atol=1e-6):
        raise ValueError(
            "Structural variant support requires explicit training_support metadata when "
            "genotypes are transformed numeric values for "
            + variant_record.variant_id
            + "."
        )
    if np.any(rounded_values < 0.0):
        raise ValueError(
            "Structural variant support requires explicit training_support metadata when "
            "raw genotypes are not non-negative counts for "
            + variant_record.variant_id
            + "."
        )
    return int(np.count_nonzero(rounded_values > 0.0))


def build_tie_map(
    genotypes: StandardizedGenotypeMatrix | np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> TieMap:
    standardized_genotypes = _as_standardized_genotypes(genotypes)
    if standardized_genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")

    exact_signature_to_group: dict[bytes, int] = {}
    sign_flipped_signature_to_group: dict[bytes, int] = {}
    tie_groups: list[TieGroup] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(standardized_genotypes.shape[1], -1, dtype=np.int32)

    for batch in standardized_genotypes.iter_column_batches(batch_size=config.genotype_batch_size):
        for local_batch_index, variant_index in enumerate(batch.variant_indices):
            standardized_column = np.where(
                batch.values[:, local_batch_index] == 0.0,
                np.float32(0.0),
                batch.values[:, local_batch_index],
            ).astype(np.float32, copy=False)
            genotype_signature = np.ascontiguousarray(standardized_column).tobytes()
            sign_flipped_column = np.where(
                standardized_column == 0.0,
                np.float32(0.0),
                -standardized_column,
            )
            sign_flipped_signature = np.ascontiguousarray(sign_flipped_column).tobytes()

            if genotype_signature in exact_signature_to_group:
                reduced_index = exact_signature_to_group[genotype_signature]
                tie_group = tie_groups[reduced_index]
                tie_group.member_indices = np.append(tie_group.member_indices, variant_index)
                tie_group.signs = np.append(tie_group.signs, 1.0)
                original_to_reduced[int(variant_index)] = reduced_index
                continue

            if genotype_signature in sign_flipped_signature_to_group:
                reduced_index = sign_flipped_signature_to_group[genotype_signature]
                tie_group = tie_groups[reduced_index]
                tie_group.member_indices = np.append(tie_group.member_indices, variant_index)
                tie_group.signs = np.append(tie_group.signs, -1.0)
                original_to_reduced[int(variant_index)] = reduced_index
                continue

            reduced_index = len(tie_groups)
            tie_groups.append(
                TieGroup(
                    representative_index=int(variant_index),
                    member_indices=np.asarray([variant_index], dtype=np.int32),
                    signs=np.asarray([1.0], dtype=np.float32),
                )
            )
            kept_variant_indices.append(int(variant_index))
            exact_signature_to_group[genotype_signature] = reduced_index
            sign_flipped_signature_to_group[sign_flipped_signature] = reduced_index
            original_to_reduced[int(variant_index)] = reduced_index

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


def _as_standardized_genotypes(
    genotypes: StandardizedGenotypeMatrix | np.ndarray,
) -> StandardizedGenotypeMatrix:
    if isinstance(genotypes, StandardizedGenotypeMatrix):
        return genotypes
    dense_genotypes = DenseRawGenotypeMatrix(np.asarray(genotypes, dtype=np.float32))
    zero_means = np.zeros(dense_genotypes.shape[1], dtype=np.float32)
    unit_scales = np.ones(dense_genotypes.shape[1], dtype=np.float32)
    return dense_genotypes.standardized(zero_means, unit_scales)

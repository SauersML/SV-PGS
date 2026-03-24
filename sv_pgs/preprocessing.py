from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord


@dataclass(slots=True)
class Preprocessor:
    means: np.ndarray
    scales: np.ndarray

    def transform(self, genotypes: np.ndarray) -> np.ndarray:
        genotype_matrix = jnp.asarray(genotypes, dtype=jnp.float32)
        imputed_genotypes = _impute_missing_values(genotype_matrix, jnp.asarray(self.means))
        return np.asarray((imputed_genotypes - self.means) / self.scales, dtype=np.float32)


def fit_preprocessor(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
) -> PreparedArrays:
    genotype_matrix = jnp.asarray(genotypes, dtype=jnp.float32)
    covariate_matrix = jnp.asarray(covariates, dtype=jnp.float32)
    target_array = jnp.asarray(targets, dtype=jnp.float32).reshape(-1)
    if genotype_matrix.ndim != 2:
        raise ValueError("genotypes must be 2D.")
    if covariate_matrix.ndim != 2:
        raise ValueError("covariates must be 2D.")
    if genotype_matrix.shape[0] != covariate_matrix.shape[0] or genotype_matrix.shape[0] != target_array.shape[0]:
        raise ValueError("genotypes, covariates, and targets must share sample dimension.")

    means = jnp.nanmean(genotype_matrix, axis=0)
    means = jnp.where(jnp.isnan(means), 0.0, means)
    imputed_genotypes = _impute_missing_values(genotype_matrix, means)
    centered_genotypes = imputed_genotypes - means
    scales = jnp.sqrt(jnp.mean(centered_genotypes * centered_genotypes, axis=0))
    scales = jnp.where(scales < config.minimum_scale, 1.0, scales)

    return PreparedArrays(
        genotypes=np.asarray(centered_genotypes / scales, dtype=np.float32),
        covariates=np.asarray(covariate_matrix, dtype=np.float32),
        targets=np.asarray(target_array, dtype=np.float32),
        means=np.asarray(means, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
    )


def _impute_missing_values(genotype_matrix: jnp.ndarray, means: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(jnp.isnan(genotype_matrix), means[None, :], genotype_matrix)


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
        support_count = _structural_variant_support(
            raw_variant_values=genotype_matrix[:, variant_index],
            variant_record=variant_record,
        )
        if support_count < config.minimum_structural_variant_carriers:
            active_flags[variant_index] = False
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
    genotypes: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> TieMap:
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be 2D.")
    if genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")

    exact_signature_to_group: dict[bytes, int] = {}
    sign_flipped_signature_to_group: dict[bytes, int] = {}
    tie_groups: list[TieGroup] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(genotypes.shape[1], -1, dtype=np.int32)

    float32_genotypes = np.ascontiguousarray(np.asarray(genotypes, dtype=np.float32))
    for variant_index in range(genotypes.shape[1]):
        standardized_column = np.where(
            float32_genotypes[:, variant_index] == 0.0,
            np.float32(0.0),
            float32_genotypes[:, variant_index],
        )
        genotype_signature = np.ascontiguousarray(standardized_column).tobytes()
        sign_flipped_signature = np.ascontiguousarray(-standardized_column).tobytes()

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

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

import sv_pgs._jax  # noqa: F401
import jax
import jax.numpy as jnp

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord, VariantStatistics
from sv_pgs.genotype import (
    DenseRawGenotypeMatrix,
    RawGenotypeBatch,
    RawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    _standardize_batch,
    as_raw_genotype_matrix,
    auto_batch_size,
)
from sv_pgs.progress import log, mem


@jax.jit
def _batch_all_stats_i8(batch_i8: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-variant statistics from int8 genotypes."""
    mask = batch_i8 != -127
    values_f32 = jnp.where(mask, batch_i8.astype(jnp.float32), 0.0)
    sums = jnp.sum(values_f32, axis=0, dtype=jnp.float64)
    counts = jnp.sum(mask, axis=0, dtype=jnp.int32)
    support = jnp.sum(mask & (batch_i8 > 0), axis=0, dtype=jnp.int32)
    safe_counts = jnp.maximum(counts, 1).astype(jnp.float32)
    means_f32 = (sums / safe_counts.astype(jnp.float64)).astype(jnp.float32)
    imputed = jnp.where(mask, values_f32, means_f32[None, :])
    centered = imputed - means_f32[None, :]
    css = jnp.sum(centered * centered, axis=0, dtype=jnp.float64)
    return sums, counts, support, css


@jax.jit
def _batch_all_stats(batch_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-variant statistics from float32 genotypes."""
    mask = ~jnp.isnan(batch_values)
    observed = jnp.where(mask, batch_values, 0.0)
    sums = jnp.sum(observed, axis=0, dtype=jnp.float64)
    counts = jnp.sum(mask, axis=0, dtype=jnp.int32)
    support = jnp.sum(mask & (jnp.abs(observed) > 0.5), axis=0, dtype=jnp.int32)
    safe_counts = jnp.maximum(counts, 1).astype(jnp.float64)
    means = sums / safe_counts
    means_f32 = means.astype(jnp.float32)
    imputed = jnp.where(mask, batch_values, means_f32[None, :])
    centered = imputed - means_f32[None, :]
    css = jnp.sum(centered * centered, axis=0, dtype=jnp.float64)
    return sums, counts, support, css


def compute_variant_statistics(
    raw_genotypes: RawGenotypeMatrix,
    config: ModelConfig,
    covariates: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> VariantStatistics:
    """Compute per-variant moments and optional marginal screening scores."""
    variant_count = raw_genotypes.shape[1]
    sample_count = raw_genotypes.shape[0]
    batch_size = auto_batch_size(sample_count)
    jax_backend = jax.default_backend()
    log(
        f"=== VARIANT STATISTICS (1-pass, JAX/{jax_backend}) ===  "
        f"{sample_count} samples x {variant_count} variants  batch_size={batch_size}  mem={mem()}"
    )

    sums = np.zeros(variant_count, dtype=np.float64)
    non_missing_counts = np.zeros(variant_count, dtype=np.int32)
    support_counts = np.zeros(variant_count, dtype=np.int32)
    centered_sum_squares = np.zeros(variant_count, dtype=np.float64)

    use_i8 = hasattr(raw_genotypes, "iter_column_batches_i8")
    if use_i8:
        log("  using int8 native path (4x less IO per batch)")
        batch_iter = raw_genotypes.iter_column_batches_i8(batch_size=batch_size)
    else:
        log(f"  using float32 path (type={type(raw_genotypes).__name__})")
        batch_iter = raw_genotypes.iter_column_batches(batch_size=batch_size)

    import time as _time

    variants_done = 0
    batch_number = 0
    for batch in batch_iter:
        batch_number += 1
        start_time = _time.monotonic()
        batch_indices = batch.variant_indices
        batch_jax = jnp.asarray(batch.values)
        if use_i8:
            batch_sums, batch_counts, batch_support, batch_css = _batch_all_stats_i8(batch_jax)
        else:
            batch_sums, batch_counts, batch_support, batch_css = _batch_all_stats(batch_jax)
        sums[batch_indices] = np.asarray(batch_sums, dtype=np.float64)
        non_missing_counts[batch_indices] = np.asarray(batch_counts, dtype=np.int32)
        support_counts[batch_indices] = np.asarray(batch_support, dtype=np.int32)
        centered_sum_squares[batch_indices] = np.asarray(batch_css, dtype=np.float64)
        del batch_jax

        batch_seconds = _time.monotonic() - start_time
        variants_done += len(batch_indices)
        if batch_number <= 3 or variants_done % max(variant_count // 10, 1) < len(batch_indices) or variants_done == variant_count:
            estimated_total = batch_seconds * ((variant_count - variants_done) / max(len(batch_indices), 1)) if batch_number >= 2 else 0.0
            log(
                f"  {variants_done}/{variant_count} ({100 * variants_done // variant_count}%)  "
                f"batch={batch_seconds:.1f}s  est_remaining={estimated_total / 60:.0f}min  mem={mem()}"
            )

    means = np.divide(
        sums,
        np.maximum(non_missing_counts, 1),
        out=np.zeros_like(sums),
        where=non_missing_counts > 0,
    ).astype(np.float32)
    allele_frequencies = np.clip(means / 2.0, 0.0, 1.0).astype(np.float32)
    scales = _scales_from_centered_sum_squares(
        centered_sum_squares=centered_sum_squares,
        sample_count=sample_count,
        minimum_scale=config.minimum_scale,
    )
    log(
        f"=== VARIANT STATISTICS DONE ===  mean_af={float(np.mean(allele_frequencies)):.4f}  "
        f"mean_scale={float(np.mean(scales)):.4f}  mean_support={float(np.mean(support_counts)):.0f}  mem={mem()}"
    )

    return VariantStatistics(
        means=means,
        scales=scales,
        allele_frequencies=allele_frequencies,
        support_counts=support_counts.astype(np.int32),
    )


def _scales_from_centered_sum_squares(
    centered_sum_squares: np.ndarray,
    sample_count: int,
    minimum_scale: float,
) -> np.ndarray:
    scales = np.sqrt(np.asarray(centered_sum_squares, dtype=np.float64) / max(int(sample_count), 1))
    return np.where(scales < minimum_scale, 1.0, scales).astype(np.float32)


@dataclass(slots=True)
class Preprocessor:
    """Stores per-variant mean and standard deviation learned during training."""

    means: np.ndarray
    scales: np.ndarray

    def transform(self, genotypes: RawGenotypeMatrix | np.ndarray) -> StandardizedGenotypeMatrix | np.ndarray:
        raw_genotypes = as_raw_genotype_matrix(genotypes)
        standardized = raw_genotypes.standardized(self.means, self.scales)
        if isinstance(genotypes, np.ndarray):
            return standardized.materialize()
        return standardized


def fit_preprocessor_from_stats(
    variant_stats: VariantStatistics,
    covariates: np.ndarray,
    targets: np.ndarray,
) -> PreparedArrays:
    """Build PreparedArrays from pre-computed variant statistics (no data passes)."""
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32).reshape(-1)
    if covariate_matrix.ndim != 2:
        raise ValueError("covariates must be 2D.")
    return PreparedArrays(
        covariates=covariate_matrix,
        targets=target_array,
        means=np.asarray(variant_stats.means, dtype=np.float32),
        scales=np.asarray(variant_stats.scales, dtype=np.float32),
    )


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
    sample_count = raw_genotypes.shape[0]
    log(f"  preprocessor: computing means and scales in single pass over {variant_count} variants...")
    sums = np.zeros(variant_count, dtype=np.float64)
    sum_squares = np.zeros(variant_count, dtype=np.float64)
    non_missing_counts = np.zeros(variant_count, dtype=np.int64)
    variants_done = 0
    for batch in raw_genotypes.iter_column_batches(batch_size=auto_batch_size(sample_count)):
        batch_jax = jnp.asarray(batch.values)
        mask = ~jnp.isnan(batch_jax)
        observed = jnp.where(mask, batch_jax, 0.0)
        batch_sums = jnp.sum(observed, axis=0)
        batch_counts = jnp.sum(mask, axis=0)
        batch_sum_sq = jnp.sum(observed * observed, axis=0)
        batch_indices = batch.variant_indices
        sums[batch_indices] = np.asarray(batch_sums, dtype=np.float64)
        non_missing_counts[batch_indices] = np.asarray(batch_counts, dtype=np.int64)
        sum_squares[batch_indices] = np.asarray(batch_sum_sq, dtype=np.float64)
        variants_done += len(batch_indices)
        if variants_done == len(batch_indices) or variants_done % max(variant_count // 10, 1) < len(batch_indices) or variants_done == variant_count:
            log(f"  preprocessor: {variants_done}/{variant_count} ({100 * variants_done // variant_count}%)  mem={mem()}")

    safe_counts = np.maximum(non_missing_counts, 1).astype(np.float64)
    means = np.where(non_missing_counts > 0, sums / safe_counts, 0.0)
    centered_sum_sq = np.maximum(sum_squares - sums * sums / safe_counts, 0.0)
    scales = _scales_from_centered_sum_squares(
        centered_sum_squares=centered_sum_sq,
        sample_count=sample_count,
        minimum_scale=config.minimum_scale,
    )
    del sums, sum_squares, non_missing_counts, safe_counts, centered_sum_sq
    log(f"  preprocessor done  mem={mem()}")

    return PreparedArrays(
        covariates=np.asarray(covariate_matrix, dtype=np.float32),
        targets=np.asarray(target_array, dtype=np.float32),
        means=np.asarray(means, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
    )


def select_active_variant_indices(
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> np.ndarray:
    """Keep every variant in the fit."""
    del config
    n_total = len(variant_records)
    log(f"  active variants: {n_total}/{n_total} kept [NO DATA PASS]")
    return np.arange(n_total, dtype=np.int32)


def build_tie_map(
    genotypes: StandardizedGenotypeMatrix | np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> TieMap:
    """Collapse exact and sign-flipped duplicate genotype columns."""
    del config
    standardized_genotypes = _as_standardized_genotypes(genotypes)
    if standardized_genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")
    n_total = standardized_genotypes.shape[1]
    log(f"  building tie map over {n_total} variants...")

    exact_signature_to_group: dict[tuple[bytes, bytes], list[int]] = {}
    sign_flipped_signature_to_group: dict[tuple[bytes, bytes], list[int]] = {}
    tie_group_member_indices: list[list[int]] = []
    tie_group_signs: list[list[float]] = []
    representative_indices: list[int] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(n_total, -1, dtype=np.int32)

    variants_done = 0
    for batch, missing_mask in _iter_tie_map_batches(standardized_genotypes):
        for local_batch_index, variant_index in enumerate(batch.variant_indices):
            record = records[int(variant_index)]
            if (
                record.variant_class in set(config.structural_variant_classes())
                and record.training_support is not None
                and int(record.training_support) <= 1
            ):
                original_to_reduced[int(variant_index)] = -1
                continue
            standardized_column = _normalize_signed_zeros(batch.values[:, local_batch_index])
            column_missing_mask = np.asarray(missing_mask[:, local_batch_index], dtype=bool)
            genotype_signature = _hashed_tie_signature(standardized_column, column_missing_mask)
            sign_flipped_signature = _hashed_tie_signature(
                _normalize_signed_zeros(-standardized_column),
                column_missing_mask,
            )

            exact_match_index = _matching_tie_group_index(
                standardized_genotypes=standardized_genotypes,
                candidate_group_indices=exact_signature_to_group.get(genotype_signature, ()),
                representative_member_indices=tie_group_member_indices,
                column=standardized_column,
                missing_mask=column_missing_mask,
                sign=1.0,
            )
            if exact_match_index is not None:
                tie_group_member_indices[exact_match_index].append(int(variant_index))
                tie_group_signs[exact_match_index].append(1.0)
                original_to_reduced[int(variant_index)] = exact_match_index
                continue

            sign_flipped_match_index = _matching_tie_group_index(
                standardized_genotypes=standardized_genotypes,
                candidate_group_indices=sign_flipped_signature_to_group.get(genotype_signature, ()),
                representative_member_indices=tie_group_member_indices,
                column=standardized_column,
                missing_mask=column_missing_mask,
                sign=-1.0,
            )
            if sign_flipped_match_index is not None:
                tie_group_member_indices[sign_flipped_match_index].append(int(variant_index))
                tie_group_signs[sign_flipped_match_index].append(-1.0)
                original_to_reduced[int(variant_index)] = sign_flipped_match_index
                continue

            reduced_index = len(representative_indices)
            representative_indices.append(int(variant_index))
            tie_group_member_indices.append([int(variant_index)])
            tie_group_signs.append([1.0])
            kept_variant_indices.append(int(variant_index))
            exact_signature_to_group.setdefault(genotype_signature, []).append(reduced_index)
            sign_flipped_signature_to_group.setdefault(sign_flipped_signature, []).append(reduced_index)
            original_to_reduced[int(variant_index)] = reduced_index

        variants_done += len(batch.variant_indices)
        if variants_done == len(batch.variant_indices) or variants_done % max(n_total // 10, 1) < len(batch.variant_indices) or variants_done == n_total:
            log(
                f"  tie map: {variants_done}/{n_total} ({100 * variants_done // n_total}%)  "
                f"unique={len(kept_variant_indices)}  groups={len(representative_indices)}  mem={mem()}"
            )

    log(f"  tie map done: {n_total} -> {len(kept_variant_indices)} unique representatives  mem={mem()}")
    return TieMap(
        kept_indices=np.asarray(kept_variant_indices, dtype=np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=[
            TieGroup(
                representative_index=representative_index,
                member_indices=np.asarray(member_indices, dtype=np.int32),
                signs=np.asarray(signs, dtype=np.float32),
            )
            for representative_index, member_indices, signs in zip(
                representative_indices,
                tie_group_member_indices,
                tie_group_signs,
                strict=True,
            )
        ],
    )


def _iter_tie_map_batches(
    standardized_genotypes: StandardizedGenotypeMatrix,
) -> Iterator[tuple[RawGenotypeBatch, np.ndarray]]:
    batch_size = auto_batch_size(standardized_genotypes.shape[0])
    if standardized_genotypes.raw is not None:
        local_start = 0
        for raw_batch in standardized_genotypes.raw.iter_column_batches(
            standardized_genotypes.variant_indices,
            batch_size=batch_size,
        ):
            local_stop = local_start + raw_batch.variant_indices.shape[0]
            local_indices = np.arange(local_start, local_stop, dtype=np.int32)
            standardized_values = _standardize_batch(
                raw_batch.values,
                standardized_genotypes.means[raw_batch.variant_indices],
                standardized_genotypes.scales[raw_batch.variant_indices],
            )
            yield RawGenotypeBatch(
                variant_indices=local_indices,
                values=standardized_values,
            ), np.isnan(raw_batch.values)
            local_start = local_stop
        return
    for batch in standardized_genotypes.iter_column_batches(batch_size=batch_size):
        yield batch, np.zeros_like(batch.values, dtype=bool)


def _normalize_signed_zeros(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.where(array == 0.0, np.float32(0.0), array).astype(np.float32, copy=False)


def _hashed_tie_signature(
    standardized_column: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[bytes, bytes]:
    value_hasher = hashlib.blake2b(digest_size=16, person=b"svpgs_value")
    value_hasher.update(np.ascontiguousarray(standardized_column).view(np.uint8))
    mask_hasher = hashlib.blake2b(digest_size=16, person=b"svpgs_mask_")
    mask_hasher.update(np.packbits(np.ascontiguousarray(missing_mask, dtype=np.uint8)))
    return value_hasher.digest(), mask_hasher.digest()


def _matching_tie_group_index(
    standardized_genotypes: StandardizedGenotypeMatrix,
    candidate_group_indices: Sequence[int],
    representative_member_indices: Sequence[list[int]],
    column: np.ndarray,
    missing_mask: np.ndarray,
    sign: float,
) -> int | None:
    for reduced_index in candidate_group_indices:
        representative_column, representative_missing_mask = _load_standardized_column_with_missingness(
            standardized_genotypes,
            int(representative_member_indices[int(reduced_index)][0]),
        )
        if not np.array_equal(missing_mask, representative_missing_mask):
            continue
        if sign < 0.0:
            representative_column = _normalize_signed_zeros(-representative_column)
        if np.array_equal(column, representative_column):
            return int(reduced_index)
    return None


def _load_standardized_column_with_missingness(
    standardized_genotypes: StandardizedGenotypeMatrix,
    local_variant_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if standardized_genotypes.raw is not None:
        raw_variant_index = int(standardized_genotypes.variant_indices[local_variant_index])
        raw_column = standardized_genotypes.raw.materialize([raw_variant_index]).reshape(-1, 1)
        standardized_column = _standardize_batch(
            raw_column,
            standardized_genotypes.means[np.asarray([raw_variant_index], dtype=np.int32)],
            standardized_genotypes.scales[np.asarray([raw_variant_index], dtype=np.int32)],
        )[:, 0]
        return _normalize_signed_zeros(standardized_column), np.isnan(raw_column[:, 0])
    if standardized_genotypes._dense_cache is None:
        dense_column = standardized_genotypes.materialize()[:, local_variant_index]
    else:
        dense_column = standardized_genotypes._dense_cache[:, local_variant_index]
    return _normalize_signed_zeros(dense_column), np.zeros(dense_column.shape[0], dtype=bool)


def collapse_tie_groups(
    records: Sequence[VariantRecord],
    tie_map: TieMap,
) -> list[VariantRecord]:
    """Create one merged record per tie group for use in the reduced model."""
    collapsed_records: list[VariantRecord] = []
    for tie_group in tie_map.reduced_to_group:
        member_records = [records[int(member_index)] for member_index in tie_group.member_indices]
        unique_variant_classes, class_membership = _class_membership(member_records)
        latent_variant_class = unique_variant_classes[0] if len(unique_variant_classes) == 1 else VariantClass.OTHER_COMPLEX_SV
        support_values = [
            float(member_record.training_support)
            for member_record in member_records
            if member_record.training_support is not None
        ]
        continuous_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_continuous_features
            }
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
                training_support=None if not support_values else int(np.round(np.mean(support_values))),
                is_repeat=any(member_record.is_repeat for member_record in member_records),
                is_copy_number=any(member_record.is_copy_number for member_record in member_records),
                prior_continuous_features={
                    feature_name: float(
                        np.mean(
                            [
                                member_record.prior_continuous_features.get(feature_name, 0.0)
                                for member_record in member_records
                            ]
                        )
                    )
                    for feature_name in continuous_feature_names
                },
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

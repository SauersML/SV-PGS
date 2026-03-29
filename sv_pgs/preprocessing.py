from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import sv_pgs._jax  # noqa: F401
import jax
import jax.numpy as jnp

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord, VariantStatistics
from sv_pgs.genotype import (
    DenseRawGenotypeMatrix,
    RawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    as_raw_genotype_matrix,
)


@jax.jit
def _batch_all_stats_i8(batch_i8: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-pass JIT operating on int8 genotypes (sentinel -127 = missing).

    Avoids float32 intermediate entirely. 4x less memory per batch.
    """
    mask = batch_i8 != -127
    values = jnp.where(mask, batch_i8.astype(jnp.float64), 0.0)

    sums = jnp.sum(values, axis=0)
    counts = jnp.sum(mask, axis=0, dtype=jnp.int32)
    support = jnp.sum(mask & (batch_i8 > 0), axis=0, dtype=jnp.int32)

    safe_counts = jnp.maximum(counts, 1).astype(jnp.float64)
    means = sums / safe_counts
    imputed = jnp.where(mask, values, means[None, :])
    centered = imputed - means[None, :]
    css = jnp.sum(centered * centered, axis=0)

    return sums, counts, support, css


@jax.jit
def _batch_all_stats_with_screening_i8(
    batch_i8: jnp.ndarray,
    residual: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-pass JIT: stats + screening on int8 genotypes."""
    mask = batch_i8 != -127
    values = jnp.where(mask, batch_i8.astype(jnp.float64), 0.0)

    sums = jnp.sum(values, axis=0)
    counts = jnp.sum(mask, axis=0, dtype=jnp.int32)
    support = jnp.sum(mask & (batch_i8 > 0), axis=0, dtype=jnp.int32)

    safe_counts = jnp.maximum(counts, 1).astype(jnp.float64)
    means = sums / safe_counts
    imputed = jnp.where(mask, values, means[None, :])
    centered = imputed - means[None, :]
    css = jnp.sum(centered * centered, axis=0)

    n = batch_i8.shape[0]
    scales = jnp.sqrt(css / jnp.maximum(n, 1))
    scales = jnp.where(scales < 1e-6, 1.0, scales)
    standardized = centered / scales[None, :]
    scores = jnp.abs(standardized.T @ residual.astype(jnp.float64))

    return sums, counts, support, css, scores


# Keep float32 versions for VCF path
@jax.jit
def _batch_all_stats(batch_values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Float32 path for VCF/dense genotypes."""
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
    """Compute all per-variant statistics in a SINGLE pass over the genotype data.

    If covariates and targets are provided, also computes marginal screening
    scores in the same pass (avoiding a separate screening pass later).
    """
    from sv_pgs.progress import log, mem
    variant_count = raw_genotypes.shape[1]
    sample_count = raw_genotypes.shape[0]
    batch_size = config.genotype_batch_size
    n_batches = (variant_count + batch_size - 1) // batch_size

    # Compute residual for screening if covariates/targets provided
    compute_scores = covariates is not None and targets is not None
    residual_jax: jnp.ndarray | None = None
    if compute_scores:
        cov = np.asarray(covariates, dtype=np.float64)
        tgt = np.asarray(targets, dtype=np.float64).reshape(-1)
        coef, *_ = np.linalg.lstsq(cov, tgt, rcond=None)
        residual = tgt - cov @ coef
        residual_jax = jnp.asarray(residual)
        log(f"=== VARIANT STATISTICS + SCREENING (1-pass, JAX) ===  {sample_count} samples x {variant_count} variants  batch_size={batch_size}  n_batches={n_batches}  mem={mem()}")
    else:
        log(f"=== VARIANT STATISTICS (1-pass, JAX) ===  {sample_count} samples x {variant_count} variants  batch_size={batch_size}  n_batches={n_batches}  mem={mem()}")

    sums = np.zeros(variant_count, dtype=np.float64)
    non_missing_counts = np.zeros(variant_count, dtype=np.int32)
    support_counts = np.zeros(variant_count, dtype=np.int32)
    centered_sum_squares = np.zeros(variant_count, dtype=np.float64)
    marginal_scores: np.ndarray | None = None
    if compute_scores:
        marginal_scores = np.zeros(variant_count, dtype=np.float64)

    # Use int8 path for PLINK data (4x less memory, avoids float32 intermediate)
    from sv_pgs.genotype import PlinkRawGenotypeMatrix
    use_i8 = isinstance(raw_genotypes, PlinkRawGenotypeMatrix)
    if use_i8:
        log(f"  using int8 native path (4x less memory per batch)")

    variants_done = 0
    batch_iter = raw_genotypes.iter_column_batches_i8(batch_size=batch_size) if use_i8 else raw_genotypes.iter_column_batches(batch_size=batch_size)
    for batch in batch_iter:
        idx = batch.variant_indices
        if use_i8:
            batch_jax = jnp.asarray(batch.values)  # int8 on JAX
            if compute_scores:
                batch_sums, batch_counts, batch_support, batch_css, batch_scores = _batch_all_stats_with_screening_i8(batch_jax, residual_jax)
                marginal_scores[idx] = np.asarray(batch_scores, dtype=np.float64)
            else:
                batch_sums, batch_counts, batch_support, batch_css = _batch_all_stats_i8(batch_jax)
        else:
            batch_jax = jnp.asarray(batch.values)  # float32 on JAX
            if compute_scores:
                batch_sums, batch_counts, batch_support, batch_css, batch_scores = _batch_all_stats_with_screening(batch_jax, residual_jax)
                marginal_scores[idx] = np.asarray(batch_scores, dtype=np.float64)
            else:
                batch_sums, batch_counts, batch_support, batch_css = _batch_all_stats(batch_jax)
        sums[idx] = np.asarray(batch_sums, dtype=np.float64)
        non_missing_counts[idx] = np.asarray(batch_counts, dtype=np.int32)
        support_counts[idx] = np.asarray(batch_support, dtype=np.int32)
        centered_sum_squares[idx] = np.asarray(batch_css, dtype=np.float64)
        del batch_jax
        variants_done += len(idx)
        if variants_done == len(idx) or variants_done % max(variant_count // 10, 1) < len(idx) or variants_done == variant_count:
            log(f"  {variants_done}/{variant_count} ({100*variants_done//variant_count}%)  mem={mem()}")

    means = np.divide(
        sums, np.maximum(non_missing_counts, 1),
        out=np.zeros_like(sums), where=non_missing_counts > 0,
    ).astype(np.float32)
    allele_frequencies = np.clip(means / 2.0, 0.0, 1.0).astype(np.float32)
    scales = np.sqrt(centered_sum_squares / max(sample_count, 1)).astype(np.float32)
    scales = np.where(scales < config.minimum_scale, 1.0, scales).astype(np.float32)
    log(f"=== VARIANT STATISTICS DONE ===  mean_af={float(np.mean(allele_frequencies)):.4f}  mean_scale={float(np.mean(scales)):.4f}  mean_support={float(np.mean(support_counts)):.0f}  mem={mem()}")

    return VariantStatistics(
        means=means,
        scales=scales,
        allele_frequencies=allele_frequencies,
        support_counts=support_counts.astype(np.int32),
        marginal_scores=marginal_scores,
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
    from sv_pgs.progress import log, mem
    raw_genotypes = as_raw_genotype_matrix(genotypes)
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32).reshape(-1)
    if covariate_matrix.ndim != 2:
        raise ValueError("covariates must be 2D.")
    if raw_genotypes.shape[0] != covariate_matrix.shape[0] or raw_genotypes.shape[0] != target_array.shape[0]:
        raise ValueError("genotypes, covariates, and targets must share sample dimension.")

    variant_count = raw_genotypes.shape[1]
    log(f"  preprocessor pass 1/2: computing means over {variant_count} variants...")
    sums = np.zeros(variant_count, dtype=np.float64)
    non_missing_counts = np.zeros(variant_count, dtype=np.int64)
    variants_done = 0
    for batch in raw_genotypes.iter_column_batches(batch_size=config.genotype_batch_size):
        mask = ~np.isnan(batch.values)
        observed = np.where(mask, batch.values, 0.0)
        sums[batch.variant_indices] = np.sum(observed, axis=0, dtype=np.float64)
        non_missing_counts[batch.variant_indices] = np.sum(mask, axis=0, dtype=np.int64)
        variants_done += len(batch.variant_indices)
        if variants_done == len(batch.variant_indices) or variants_done % max(variant_count // 10, 1) < len(batch.variant_indices) or variants_done == variant_count:
            log(f"  preprocessor pass 1/2: {variants_done}/{variant_count} ({100*variants_done//variant_count}%)  mem={mem()}")

    means = np.divide(
        sums,
        np.maximum(non_missing_counts, 1),
        out=np.zeros_like(sums),
        where=non_missing_counts > 0,
    )

    log(f"  preprocessor pass 2/2: computing scales over {variant_count} variants...")
    centered_sum_squares = np.zeros(variant_count, dtype=np.float64)
    variants_done = 0
    for batch in raw_genotypes.iter_column_batches(batch_size=config.genotype_batch_size):
        batch_means = means[batch.variant_indices]
        imputed = np.where(np.isnan(batch.values), batch_means[None, :], batch.values)
        centered = imputed - batch_means[None, :]
        centered_sum_squares[batch.variant_indices] = np.sum(centered * centered, axis=0, dtype=np.float64)
        variants_done += len(batch.variant_indices)
        if variants_done == len(batch.variant_indices) or variants_done % max(variant_count // 10, 1) < len(batch.variant_indices) or variants_done == variant_count:
            log(f"  preprocessor pass 2/2: {variants_done}/{variant_count} ({100*variants_done//variant_count}%)  mem={mem()}")

    scales = np.sqrt(centered_sum_squares / max(raw_genotypes.shape[0], 1))
    scales = np.where(scales < config.minimum_scale, 1.0, scales)
    log(f"  preprocessor done  mem={mem()}")

    return PreparedArrays(
        covariates=np.asarray(covariate_matrix, dtype=np.float32),
        targets=np.asarray(target_array, dtype=np.float32),
        means=np.asarray(means, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
    )


def select_active_variant_indices(
    variant_records: Sequence[VariantRecord],
    support_counts: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    """Select active variants using pre-computed support counts (no data pass needed)."""
    from sv_pgs.progress import log
    n_total = len(variant_records)
    active_flags = np.ones(n_total, dtype=bool)
    structural_variant_classes = set(config.structural_variant_classes())
    sv_checked = 0
    sv_filtered = 0
    for variant_index, variant_record in enumerate(variant_records):
        if variant_record.variant_class not in structural_variant_classes:
            continue
        sv_checked += 1
        support = int(support_counts[variant_index])
        if variant_record.training_support is not None:
            support = variant_record.training_support
        if support < config.minimum_structural_variant_carriers:
            active_flags[variant_index] = False
            sv_filtered += 1
    result = np.where(active_flags)[0].astype(np.int32)
    log(f"  active variants: {len(result)}/{n_total} kept (SVs checked={sv_checked} filtered={sv_filtered}) [NO DATA PASS]")
    return result


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
    from sv_pgs.progress import log, mem
    standardized_genotypes = _as_standardized_genotypes(genotypes)
    if standardized_genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")
    n_total = standardized_genotypes.shape[1]
    log(f"  building tie map over {n_total} variants...")

    exact_signature_to_group: dict[bytes, int] = {}
    sign_flipped_signature_to_group: dict[bytes, int] = {}
    tie_group_member_indices: list[list[int]] = []
    tie_group_signs: list[list[float]] = []
    representative_indices: list[int] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(n_total, -1, dtype=np.int32)

    variants_done = 0
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
                tie_group_member_indices[reduced_index].append(int(variant_index))
                tie_group_signs[reduced_index].append(1.0)
                original_to_reduced[int(variant_index)] = reduced_index
                continue

            if genotype_signature in sign_flipped_signature_to_group:
                reduced_index = sign_flipped_signature_to_group[genotype_signature]
                tie_group_member_indices[reduced_index].append(int(variant_index))
                tie_group_signs[reduced_index].append(-1.0)
                original_to_reduced[int(variant_index)] = reduced_index
                continue

            reduced_index = len(representative_indices)
            representative_indices.append(int(variant_index))
            tie_group_member_indices.append([int(variant_index)])
            tie_group_signs.append([1.0])
            kept_variant_indices.append(int(variant_index))
            exact_signature_to_group[genotype_signature] = reduced_index
            sign_flipped_signature_to_group[sign_flipped_signature] = reduced_index
            original_to_reduced[int(variant_index)] = reduced_index
        variants_done += len(batch.variant_indices)
        if variants_done == len(batch.variant_indices) or variants_done % max(n_total // 10, 1) < len(batch.variant_indices) or variants_done == n_total:
            log(f"  tie map: {variants_done}/{n_total} ({100*variants_done//n_total}%)  unique={len(kept_variant_indices)}  groups={len(representative_indices)}  mem={mem()}")

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

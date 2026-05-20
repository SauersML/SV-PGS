from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Hashable, Iterator, Sequence, TypeVar, cast

import numpy as np

import sv_pgs._jax as _jax_side_effects  # side-effect: configures JAX/XLA env
del _jax_side_effects
import jax
import jax.numpy as jnp

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord, VariantStatistics
from sv_pgs.genotype import (
    DenseRawGenotypeMatrix,
    Int8BatchCapable,
    RawGenotypeBatch,
    RawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    _supports_int8_batches,
    _standardize_batch,
    as_raw_genotype_matrix,
    auto_batch_size,
    auto_batch_size_i8,
)
from sv_pgs.plink import PLINK_MISSING_INT8
from sv_pgs.progress import log, mem

HARD_CALL_TIE_SIGNATURE_TARGET_BYTES = 256_000_000
_HARD_CALL_EXACT_SIGNATURE_LUT = np.asarray(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 3],
        [0, 0, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 1, 3],
        [0, 1, 2, 3],
    ],
    dtype=np.int8,
)
_HARD_CALL_SIGN_FLIPPED_SIGNATURE_LUT = np.asarray(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 3],
        [0, 0, 0, 0],
        [1, 0, 0, 3],
        [0, 1, 0, 3],
        [2, 1, 0, 3],
    ],
    dtype=np.int8,
)

@jax.jit
def _batch_all_stats_i8(batch_i8: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-variant statistics from int8 genotypes."""
    mask = batch_i8 != PLINK_MISSING_INT8
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
) -> VariantStatistics:
    """Compute per-variant moments from a single streaming pass."""
    variant_count = raw_genotypes.shape[1]
    sample_count = raw_genotypes.shape[0]
    jax_backend = jax.default_backend()
    batch_size = auto_batch_size_i8(sample_count) if _supports_int8_batches(raw_genotypes) else auto_batch_size(sample_count)
    log(
        f"=== VARIANT STATISTICS (1-pass, JAX/{jax_backend}) ===  "
        f"{sample_count} samples x {variant_count} variants  batch_size={batch_size}  mem={mem()}"
    )
    sums = np.zeros(variant_count, dtype=np.float64)
    non_missing_counts = np.zeros(variant_count, dtype=np.int32)
    support_counts = np.zeros(variant_count, dtype=np.int32)
    centered_sum_squares = np.zeros(variant_count, dtype=np.float64)
    dosage_like = np.ones(variant_count, dtype=bool)

    use_i8 = _supports_int8_batches(raw_genotypes)
    if use_i8:
        log("  using int8 native path (4x less IO per batch)")
        batch_iter = cast(Int8BatchCapable, raw_genotypes).iter_column_batches_i8(batch_size=batch_size)
    else:
        log(f"  using float32 path (type={type(raw_genotypes).__name__})")
        batch_iter = raw_genotypes.iter_column_batches(batch_size=batch_size)

    import time as _time

    variants_done = 0
    batch_number = 0
    cumulative_fetch_seconds = 0.0
    cumulative_compute_seconds = 0.0
    overall_start = _time.monotonic()
    iter_handle = iter(batch_iter)
    while True:
        # Measure the fetch (next-batch) time separately from the compute
        # time. The fetch path is where streaming-source bottlenecks
        # (mmap page faults, bed decode, prefetch contention) show up;
        # the compute path is JAX kernel + D2H copy. Without splitting
        # them, a sudden disk stall and a JAX recompile look identical.
        fetch_start = _time.monotonic()
        try:
            batch = next(iter_handle)
        except StopIteration:
            break
        fetch_seconds = _time.monotonic() - fetch_start

        compute_start = _time.monotonic()
        batch_number += 1
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
        if not use_i8:
            batch_values = np.asarray(batch.values, dtype=np.float32)
            observed_mask = ~np.isnan(batch_values)
            bounded = np.all(
                (~observed_mask) | ((batch_values >= 0.0) & (batch_values <= 2.0)),
                axis=0,
            )
            dosage_like[batch_indices] = bounded
        del batch_jax
        compute_seconds = _time.monotonic() - compute_start

        cumulative_fetch_seconds += fetch_seconds
        cumulative_compute_seconds += compute_seconds
        variants_done += len(batch_indices)
        # Every batch logs now (was: every ~10%). Cheap and exposes
        # per-batch I/O stalls that the periodic-only log would average
        # away. The line stays short — one batch per line.
        wall_seconds = fetch_seconds + compute_seconds
        avg_wall = (cumulative_fetch_seconds + cumulative_compute_seconds) / max(batch_number, 1)
        remaining_batches = max((variant_count - variants_done) + len(batch_indices) - 1, 0) // max(len(batch_indices), 1)
        eta_seconds = avg_wall * remaining_batches
        log(
            f"  batch {batch_number} variants={variants_done}/{variant_count} "
            f"({100 * variants_done // variant_count}%)  "
            f"fetch={fetch_seconds:.2f}s compute={compute_seconds:.2f}s wall={wall_seconds:.2f}s  "
            f"avg_wall={avg_wall:.2f}s  eta={eta_seconds/60:.1f}min  mem={mem()}"
        )

    total_seconds = _time.monotonic() - overall_start
    log(
        f"  variant-stats pass done: {batch_number} batches in {total_seconds:.1f}s  "
        f"(fetch_total={cumulative_fetch_seconds:.1f}s, "
        f"compute_total={cumulative_compute_seconds:.1f}s)  mem={mem()}"
    )

    means = np.divide(
        sums,
        np.maximum(non_missing_counts, 1),
        out=np.zeros_like(sums),
        where=non_missing_counts > 0,
    ).astype(np.float32)
    allele_frequencies = _allele_frequencies_from_means(means=means, dosage_like=dosage_like)
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


def compute_marginal_z_scores(
    standardized_genotypes: StandardizedGenotypeMatrix,
    active_variant_indices: np.ndarray,
    covariate_matrix: np.ndarray,
    target_vector: np.ndarray,
) -> np.ndarray:
    """Marginal univariate z-scores, residualized on covariates.

    For each active variant j:
        beta_marg_j = X_j_std^T y_resid / n
        z_j = X_j_std^T y_resid / sqrt(n * sigma2_resid)
    where y_resid is the residual of y after the OLS projection onto the
    covariate columns. Under the null hypothesis of no marginal association
    (after covariate adjustment) and well-behaved standardization, z_j is
    approximately N(0, 1). For binary y the linearized z is a Rao-score-style
    statistic with the same asymptotic null distribution.

    The returned array is aligned with `active_variant_indices` (one z-score
    per active variant). Used by the budget-aware variant cap to drop noise
    variants that would otherwise force the slow stochastic-block fallback.
    """
    target_f64 = np.asarray(target_vector, dtype=np.float64).reshape(-1)
    n = target_f64.shape[0]
    if active_variant_indices.size == 0 or n == 0:
        return np.zeros(active_variant_indices.shape[0], dtype=np.float32)

    covariate_f64 = np.asarray(covariate_matrix, dtype=np.float64)
    if covariate_f64.ndim != 2 or covariate_f64.shape[0] != n:
        raise ValueError(
            f"covariate_matrix must have shape ({n}, _); got {covariate_f64.shape}"
        )

    if covariate_f64.shape[1] == 0:
        y_resid = target_f64 - float(target_f64.mean())
    else:
        # Solve the small covariate OLS problem (n × p_cov, p_cov is typically
        # < 30). lstsq handles rank-deficient designs (e.g., redundant
        # intercept) gracefully.
        alpha_cov, _, _, _ = np.linalg.lstsq(covariate_f64, target_f64, rcond=None)
        y_resid = target_f64 - covariate_f64 @ alpha_cov

    sigma2_resid = float(np.dot(y_resid, y_resid) / max(n, 1))
    if sigma2_resid <= 0.0:
        return np.zeros(active_variant_indices.shape[0], dtype=np.float32)

    # X_std^T y_resid over the active variants only — one streaming pass via
    # the existing transpose_matvec infrastructure.
    active_subset = standardized_genotypes.subset(np.asarray(active_variant_indices, dtype=np.int32))
    sum_xy = np.asarray(
        active_subset.transpose_matvec_numpy(y_resid.astype(np.float32)),
        dtype=np.float64,
    ).reshape(-1)

    z_denominator = float(np.sqrt(float(n) * sigma2_resid))
    if z_denominator <= 0.0:
        return np.zeros(active_variant_indices.shape[0], dtype=np.float32)
    return (sum_xy / z_denominator).astype(np.float32)


def _allele_frequencies_from_means(
    means: np.ndarray,
    dosage_like: np.ndarray,
) -> np.ndarray:
    mean_array = np.asarray(means, dtype=np.float32)
    dosage_mask = np.asarray(dosage_like, dtype=bool)
    return np.where(
        dosage_mask,
        np.clip(mean_array / 2.0, 0.0, 1.0),
        np.float32(0.5),
    ).astype(np.float32, copy=False)


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
        support_counts=np.asarray(variant_stats.support_counts, dtype=np.int32),
    )


def fit_preprocessor(
    genotypes: RawGenotypeMatrix | np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
) -> PreparedArrays:
    raw_genotypes = as_raw_genotype_matrix(genotypes)
    variant_stats = compute_variant_statistics(raw_genotypes=raw_genotypes, config=config)
    return fit_preprocessor_from_stats(
        variant_stats=variant_stats,
        covariates=covariates,
        targets=targets,
    )


# ---------------------------------------------------------------------------
# Content-keyed disk cache for the MAF filter and the tie map.
#
# Both `select_active_variant_indices` and `build_tie_map` produce deterministic
# outputs that take minutes on biobank-scale inputs. Persisting the result keyed
# by a content hash makes the second run near-instant. The cache is opt-in via
# a `cache_dir` parameter (no environment variable, per SPEC.md). A missing
# cache file is treated as a cold miss; a corrupt or mismatched cache file is
# treated as a hard error (per SPEC.md: never silently swallow failures).
# ---------------------------------------------------------------------------

_PREPROCESSING_CACHE_VERSION = 1
_MAF_CACHE_FILE_PREFIX = "maf_filter"
_TIE_CACHE_FILE_PREFIX = "tie_map"


def _hash_variant_records_for_maf(variant_records: Sequence[VariantRecord]) -> bytes:
    """Stable digest of the fields `select_active_variant_indices` reads."""
    hasher = hashlib.sha256()
    hasher.update(f"variant_records:{len(variant_records)}:".encode("utf-8"))
    # Pack allele_frequency into a numpy array so the bytes are well-defined.
    allele_frequencies = np.asarray(
        [float(record.allele_frequency) for record in variant_records],
        dtype=np.float64,
    )
    hasher.update(allele_frequencies.tobytes())
    # Variant identity (variant_id) anchors the digest to the specific dataset
    # so that two datasets with the same allele frequencies do not collide.
    variant_id_blob = "\x00".join(str(record.variant_id) for record in variant_records).encode("utf-8")
    hasher.update(hashlib.sha256(variant_id_blob).digest())
    return hasher.digest()


def _maf_cache_key(
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"v{_PREPROCESSING_CACHE_VERSION}:maf:".encode("utf-8"))
    hasher.update(_hash_variant_records_for_maf(variant_records))
    hasher.update(
        f"min_maf={config.minimum_minor_allele_frequency:.17g}".encode("utf-8")
    )
    return hasher.hexdigest()[:24]


def _maf_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{_MAF_CACHE_FILE_PREFIX}.{cache_key}.npz"


def _load_maf_filter_from_cache(cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cached_arrays:
        cached_indices = np.asarray(cached_arrays["active_variant_indices"], dtype=np.int32)
    if cached_indices.ndim != 1:
        raise ValueError(
            f"MAF filter cache at {cache_path} is corrupt: indices must be 1D, got shape {cached_indices.shape}"
        )
    return cached_indices


def _save_maf_filter_to_cache(cache_path: Path, active_variant_indices: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends `.npz` if the filename lacks it, so the temp file must
    # end in `.npz` to land where we expect.
    tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez(
        tmp_path,
        active_variant_indices=np.asarray(active_variant_indices, dtype=np.int32),
    )
    tmp_path.replace(cache_path)


def select_active_variant_indices(
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
    *,
    cache_dir: Path | None = None,
) -> np.ndarray:
    n_total = len(variant_records)
    if n_total == 0:
        return np.zeros(0, dtype=np.int32)

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_key = _maf_cache_key(variant_records, config)
        cache_path = _maf_cache_path(Path(cache_dir), cache_key)
        cached_indices = _load_maf_filter_from_cache(cache_path)
        if cached_indices is not None:
            log(
                f"  active variants: {cached_indices.shape[0]}/{n_total} restored from MAF cache "
                + f"(min_maf={config.minimum_minor_allele_frequency:.6f}, key={cache_key})"
            )
            return cached_indices

    maf_kept = np.asarray(
        [
            variant_index
            for variant_index, record in enumerate(variant_records)
            if _minor_allele_frequency(record.allele_frequency) >= config.minimum_minor_allele_frequency
        ],
        dtype=np.int32,
    )
    if maf_kept.shape[0] == 0:
        log(
            f"  active variants: 0/{n_total} kept after MAF filter "
            + f"(min_maf={config.minimum_minor_allele_frequency:.6f})"
        )
    else:
        log(
            f"  active variants: {maf_kept.shape[0]}/{n_total} kept after MAF filter "
            + f"(min_maf={config.minimum_minor_allele_frequency:.6f})"
        )

    if cache_path is not None:
        _save_maf_filter_to_cache(cache_path, maf_kept)

    return maf_kept


def _minor_allele_frequency(allele_frequency: float) -> float:
    normalized_frequency = float(np.clip(allele_frequency, 0.0, 1.0))
    return min(normalized_frequency, 1.0 - normalized_frequency)


_TIE_MAP_POSITION_WINDOW = 100_000  # 100 KB — duplicate SV calls are always nearby


def _empty_tie_map(original_variant_count: int) -> TieMap:
    original_to_reduced = np.full(original_variant_count, -1, dtype=np.int32)
    return TieMap(
        kept_indices=np.zeros(0, dtype=np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=[],
    )


def _compact_identity_tie_map(variant_count: int) -> TieMap:
    identity_indices = np.arange(int(variant_count), dtype=np.int32)
    return TieMap(
        kept_indices=identity_indices,
        original_to_reduced=identity_indices.copy(),
        reduced_to_group=[],
    )


def _build_tie_map_windowed(
    standardized_genotypes: StandardizedGenotypeMatrix,
    records: Sequence[VariantRecord],
) -> TieMap | None:
    """Fast tie map using (chromosome, support_count, position) pre-filter.

    Two variant columns can only be identical if they have the same carrier count
    on the same chromosome near the same position. This pre-filter eliminates 99%+
    of comparisons, reading genotype data only for the tiny number of candidate pairs.
    Returns None if pre-filtering isn't possible (no support counts or records).
    """
    n_total = standardized_genotypes.shape[1]
    if n_total == 0:
        return _empty_tie_map(0)

    support_counts = standardized_genotypes.support_counts
    if support_counts is None:
        return None  # can't pre-filter without support counts
    if standardized_genotypes.raw is None or not _supports_int8_batches(standardized_genotypes.raw):
        return None  # windowed pruning is only safe on raw hardcall int8 columns

    sample_count = standardized_genotypes.shape[0]
    variant_indices = standardized_genotypes.variant_indices
    selected_support = support_counts[variant_indices].astype(np.int64)

    # Group by (chromosome, support_count) — identical columns must match both.
    # Also group by (chromosome, N - support_count) to catch sign-flipped pairs.
    from collections import defaultdict
    exact_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        exact_groups[(record.chromosome, int(selected_support[i]))].append(i)

    # Find candidate pairs: same (chr, support) AND within position window
    candidate_pairs: list[tuple[int, int, float]] = []  # (i, j, sign)
    for (chrom, _sup), indices in exact_groups.items():
        if len(indices) < 2:
            continue
        # Sort by position within group
        indices_sorted = sorted(indices, key=lambda i: records[i].position)
        positions = [records[i].position for i in indices_sorted]
        for a in range(len(indices_sorted)):
            for b in range(a + 1, len(indices_sorted)):
                if positions[b] - positions[a] > _TIE_MAP_POSITION_WINDOW:
                    break
                candidate_pairs.append((indices_sorted[a], indices_sorted[b], 1.0))

    # Also check sign-flipped pairs: support_count_a + support_count_b ≈ N
    # (one column is the negation of the other)
    flip_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        flip_key = (record.chromosome, int(min(selected_support[i], sample_count - selected_support[i])))
        flip_groups[flip_key].append(i)
    for (chrom, _), indices in flip_groups.items():
        if len(indices) < 2:
            continue
        indices_sorted = sorted(indices, key=lambda i: records[i].position)
        positions = [records[i].position for i in indices_sorted]
        for a in range(len(indices_sorted)):
            for b in range(a + 1, len(indices_sorted)):
                if positions[b] - positions[a] > _TIE_MAP_POSITION_WINDOW:
                    break
                sa = int(selected_support[indices_sorted[a]])
                sb = int(selected_support[indices_sorted[b]])
                if sa == sb:
                    continue  # already checked in exact_groups
                candidate_pairs.append((indices_sorted[a], indices_sorted[b], -1.0))

    log(
        f"  tie map windowed pre-filter: {len(candidate_pairs)} candidate pairs "
        f"from {n_total} variants  mem={mem()}"
    )

    if not candidate_pairs:
        # No candidates — every variant is unique
        return _compact_identity_tie_map(n_total)

    # Read genotype columns ONLY for candidates and compare
    raw_int8 = standardized_genotypes.raw
    has_int8_batches = raw_int8 is not None and _supports_int8_batches(raw_int8)
    # Collect unique variant indices we need to read
    needed_local_indices = sorted({i for pair in candidate_pairs for i in (pair[0], pair[1])})
    column_cache: dict[int, np.ndarray] = {}
    if has_int8_batches:
        from typing import cast as _cast
        raw_typed = _cast(Int8BatchCapable, raw_int8)
        raw_variant_indices = np.asarray([int(variant_indices[i]) for i in needed_local_indices], dtype=np.int32)
        batch_size = max(min(len(raw_variant_indices), 512), 1)
        col_idx = 0
        for raw_batch in raw_typed.iter_column_batches_i8(raw_variant_indices, batch_size=batch_size):
            batch_values = np.asarray(raw_batch.values, dtype=np.int8)
            for j in range(batch_values.shape[1]):
                column_cache[needed_local_indices[col_idx]] = batch_values[:, j].copy()
                col_idx += 1
    else:
        # Fallback: read from standardized float path
        for local_idx in needed_local_indices:
            col = standardized_genotypes.subset(
                np.array([local_idx], dtype=np.int32)
            ).materialize()
            column_cache[local_idx] = np.asarray(col[:, 0], dtype=np.float32)

    log(f"  read {len(column_cache)} columns for candidate comparison  mem={mem()}")

    # Union-find for merging tied variants
    parent = list(range(n_total))
    sign_to_root = [1.0] * n_total

    def find(x: int) -> tuple[int, float]:
        sign = 1.0
        while parent[x] != x:
            sign *= sign_to_root[x]
            x = parent[x]
        return x, sign

    def union(a: int, b: int, sign: float) -> None:
        ra, sa = find(a)
        rb, sb = find(b)
        if ra == rb:
            return
        # Merge b's root into a's root
        parent[rb] = ra
        sign_to_root[rb] = sign * sa * sb

    # Check each candidate pair
    ties_found = 0
    for i, j, expected_sign in candidate_pairs:
        col_i = column_cache[i]
        col_j = column_cache[j]
        if expected_sign > 0:
            if np.array_equal(col_i, col_j):
                union(i, j, 1.0)
                ties_found += 1
        else:
            # Check if col_i == -col_j (with missing values handled)
            # For int8: missing = -1, so only compare non-missing
            if col_i.dtype == np.int8:
                valid = (col_i != -1) & (col_j != -1)
                if np.all(col_i[valid] == -col_j[valid]) and valid.sum() > 0:
                    union(i, j, -1.0)
                    ties_found += 1
            else:
                if np.allclose(col_i, -col_j, atol=1e-6):
                    union(i, j, -1.0)
                    ties_found += 1

    log(f"  tie map: {ties_found} ties found among {len(candidate_pairs)} candidates")
    if ties_found == 0:
        log(f"  tie map done: {n_total} -> {n_total} unique  (0 ties collapsed)  mem={mem()}")
        return _compact_identity_tie_map(n_total)

    # Build TieMap from union-find
    group_map: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i in range(n_total):
        root, sign = find(i)
        group_map[root].append((i, sign))

    kept_indices: list[int] = []
    original_to_reduced = np.full(n_total, -1, dtype=np.int32)
    reduced_to_group: list[TieGroup] = []
    for reduced_idx, (root, members) in enumerate(sorted(group_map.items())):
        kept_indices.append(root)
        member_indices = np.array([m[0] for m in members], dtype=np.int32)
        signs = np.array([m[1] for m in members], dtype=np.float32)
        for m_idx, _sign in members:
            original_to_reduced[m_idx] = reduced_idx
        reduced_to_group.append(TieGroup(
            representative_index=root,
            member_indices=member_indices,
            signs=signs,
        ))

    log(
        f"  tie map done: {n_total} -> {len(kept_indices)} unique  "
        f"({ties_found} ties collapsed)  mem={mem()}"
    )
    return TieMap(
        kept_indices=np.asarray(kept_indices, dtype=np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=reduced_to_group,
    )


def _hash_standardized_genotypes_for_tie_map(
    standardized_genotypes: StandardizedGenotypeMatrix,
) -> bytes:
    """Digest the per-variant view of a standardized matrix.

    The tie map depends on (a) which raw columns are selected, (b) the
    standardization parameters that turn raw columns into standardized
    columns, and (c) the sample count. All three are captured by the small
    per-variant arrays carried on the StandardizedGenotypeMatrix; the raw
    sample-by-variant array is not hashed because it would be prohibitively
    large and the per-variant statistics are themselves derived from it.
    """
    hasher = hashlib.sha256()
    sample_count, variant_count = standardized_genotypes.shape
    hasher.update(f"std:{sample_count}x{variant_count}:".encode("utf-8"))
    variant_indices = np.asarray(standardized_genotypes.variant_indices, dtype=np.int64)
    hasher.update(variant_indices.tobytes())
    means = np.asarray(standardized_genotypes.means, dtype=np.float32)
    scales = np.asarray(standardized_genotypes.scales, dtype=np.float32)
    hasher.update(means.tobytes())
    hasher.update(scales.tobytes())
    support_counts = standardized_genotypes.support_counts
    if support_counts is None:
        hasher.update(b"support=none")
    else:
        support_array = np.asarray(support_counts, dtype=np.int32)
        hasher.update(b"support=")
        hasher.update(support_array.tobytes())
    return hasher.digest()


def _hash_records_for_tie_map(records: Sequence[VariantRecord]) -> bytes:
    """Digest the record fields the windowed tie-map pre-filter reads."""
    hasher = hashlib.sha256()
    hasher.update(f"records:{len(records)}:".encode("utf-8"))
    chromosomes_blob = "\x00".join(str(record.chromosome) for record in records).encode("utf-8")
    hasher.update(hashlib.sha256(chromosomes_blob).digest())
    positions = np.asarray([int(record.position) for record in records], dtype=np.int64)
    hasher.update(positions.tobytes())
    variant_id_blob = "\x00".join(str(record.variant_id) for record in records).encode("utf-8")
    hasher.update(hashlib.sha256(variant_id_blob).digest())
    return hasher.digest()


def _tie_map_cache_key(
    standardized_genotypes: StandardizedGenotypeMatrix,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"v{_PREPROCESSING_CACHE_VERSION}:tie:".encode("utf-8"))
    hasher.update(_hash_standardized_genotypes_for_tie_map(standardized_genotypes))
    hasher.update(_hash_records_for_tie_map(records))
    hasher.update(
        f"min_maf={config.minimum_minor_allele_frequency:.17g}|"
        f"min_scale={config.minimum_scale:.17g}|"
        f"tie_window={_TIE_MAP_POSITION_WINDOW}".encode("utf-8")
    )
    return hasher.hexdigest()[:24]


def _tie_map_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{_TIE_CACHE_FILE_PREFIX}.{cache_key}.npz"


def _serialize_tie_map(tie_map: TieMap) -> dict[str, np.ndarray]:
    group_count = len(tie_map.reduced_to_group)
    representative_indices = np.asarray(
        [int(group.representative_index) for group in tie_map.reduced_to_group],
        dtype=np.int32,
    )
    member_lengths = np.asarray(
        [int(group.member_indices.shape[0]) for group in tie_map.reduced_to_group],
        dtype=np.int32,
    )
    group_starts = np.zeros(group_count + 1, dtype=np.int64)
    if group_count > 0:
        np.cumsum(member_lengths.astype(np.int64), out=group_starts[1:])
    flat_size = int(group_starts[-1]) if group_count > 0 else 0
    flat_member_indices = np.empty(flat_size, dtype=np.int32)
    flat_signs = np.empty(flat_size, dtype=np.float32)
    for group_index, group in enumerate(tie_map.reduced_to_group):
        start = int(group_starts[group_index])
        stop = int(group_starts[group_index + 1])
        flat_member_indices[start:stop] = np.asarray(group.member_indices, dtype=np.int32)
        flat_signs[start:stop] = np.asarray(group.signs, dtype=np.float32)
    return {
        "kept_indices": np.asarray(tie_map.kept_indices, dtype=np.int32),
        "original_to_reduced": np.asarray(tie_map.original_to_reduced, dtype=np.int32),
        "representative_indices": representative_indices,
        "group_starts": group_starts,
        "flat_member_indices": flat_member_indices,
        "flat_signs": flat_signs,
    }


def _deserialize_tie_map(cached_arrays: dict[str, np.ndarray] | "np.lib.npyio.NpzFile") -> TieMap:
    kept_indices = np.asarray(cached_arrays["kept_indices"], dtype=np.int32)
    original_to_reduced = np.asarray(cached_arrays["original_to_reduced"], dtype=np.int32)
    representative_indices = np.asarray(cached_arrays["representative_indices"], dtype=np.int32)
    group_starts = np.asarray(cached_arrays["group_starts"], dtype=np.int64)
    flat_member_indices = np.asarray(cached_arrays["flat_member_indices"], dtype=np.int32)
    flat_signs = np.asarray(cached_arrays["flat_signs"], dtype=np.float32)
    group_count = representative_indices.shape[0]
    if group_starts.shape[0] != group_count + 1:
        raise ValueError(
            f"tie map cache is corrupt: group_starts length {group_starts.shape[0]} "
            f"does not match representative count {group_count} + 1"
        )
    reduced_to_group: list[TieGroup] = []
    for group_index in range(group_count):
        start = int(group_starts[group_index])
        stop = int(group_starts[group_index + 1])
        reduced_to_group.append(
            TieGroup(
                representative_index=int(representative_indices[group_index]),
                member_indices=flat_member_indices[start:stop].copy(),
                signs=flat_signs[start:stop].copy(),
            )
        )
    return TieMap(
        kept_indices=kept_indices,
        original_to_reduced=original_to_reduced,
        reduced_to_group=reduced_to_group,
    )


def _load_tie_map_from_cache(cache_path: Path) -> TieMap | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as cached_arrays:
        return _deserialize_tie_map(cached_arrays)


def _save_tie_map_to_cache(cache_path: Path, tie_map: TieMap) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez(tmp_path, **_serialize_tie_map(tie_map))
    tmp_path.replace(cache_path)


def build_tie_map(
    genotypes: StandardizedGenotypeMatrix | np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
    *,
    cache_dir: Path | None = None,
) -> TieMap:
    """Collapse exact and sign-flipped duplicate genotype columns."""
    standardized_genotypes = _as_standardized_genotypes(genotypes)
    if standardized_genotypes.shape[1] != len(records):
        raise ValueError("genotypes and records length mismatch.")
    n_total = standardized_genotypes.shape[1]

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_key = _tie_map_cache_key(standardized_genotypes, records, config)
        cache_path = _tie_map_cache_path(Path(cache_dir), cache_key)
        cached_tie_map = _load_tie_map_from_cache(cache_path)
        if cached_tie_map is not None:
            log(
                f"  tie map restored from cache: "
                f"{n_total} -> {len(cached_tie_map.kept_indices)} unique (key={cache_key})"
            )
            return cached_tie_map

    tie_map = _build_tie_map_uncached(standardized_genotypes, records)

    if cache_path is not None:
        _save_tie_map_to_cache(cache_path, tie_map)

    return tie_map


def _build_tie_map_uncached(
    standardized_genotypes: StandardizedGenotypeMatrix,
    records: Sequence[VariantRecord],
) -> TieMap:
    n_total = standardized_genotypes.shape[1]
    log(f"  building tie map over {n_total} variants...")

    # Try fast windowed pre-filter first (avoids full data pass)
    windowed_result = _build_tie_map_windowed(standardized_genotypes, records)
    if windowed_result is not None:
        return windowed_result

    using_raw_int8_fast_path = bool(
        standardized_genotypes.raw is not None
        and _supports_int8_batches(standardized_genotypes.raw)
        and not _uses_identity_standardization(standardized_genotypes)
    )
    if using_raw_int8_fast_path:
        log("  tie map fast path: hardcall int8 canonical signatures")
        return _build_tie_map_from_hardcall_int8(standardized_genotypes)

    exact_signature_to_group: dict[tuple[bytes, bytes], list[int]] = {}
    sign_flipped_signature_to_group: dict[tuple[bytes, bytes], list[int]] = {}
    tie_group_member_indices: list[list[int]] = []
    tie_group_signs: list[list[float]] = []
    representative_indices: list[int] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(n_total, -1, dtype=np.int32)

    using_raw_batches = standardized_genotypes.raw is not None and not _uses_identity_standardization(standardized_genotypes)
    variants_done = 0
    for batch, missing_mask in _iter_tie_map_batches(standardized_genotypes):
        for local_batch_index, variant_index in enumerate(batch.variant_indices):
            column_missing_mask = np.asarray(missing_mask[:, local_batch_index], dtype=bool)
            use_rank_canonicalization = using_raw_batches and _should_use_rank_canonicalization(
                batch.values[:, local_batch_index],
                column_missing_mask,
            )
            if use_rank_canonicalization:
                rank_source = np.rint(np.asarray(batch.values[:, local_batch_index], dtype=np.float32)).astype(np.int8, copy=False)
                tie_column = _canonicalize_tie_column(
                    rank_source,
                    column_missing_mask,
                )
                sign_flipped_column = _sign_flip_canonical_tie_column(tie_column)
            else:
                if using_raw_batches:
                    tie_column = _standardize_tie_column_from_raw(
                        standardized_genotypes,
                        batch.values[:, local_batch_index],
                        int(variant_index),
                    )
                else:
                    tie_column = _normalize_signed_zeros(batch.values[:, local_batch_index])
                sign_flipped_column = _normalize_signed_zeros(-tie_column)

            genotype_signature = _hashed_tie_signature(tie_column, column_missing_mask)
            sign_flipped_signature = _hashed_tie_signature(sign_flipped_column, column_missing_mask)

            exact_match_index = _matching_tie_group_index(
                standardized_genotypes=standardized_genotypes,
                candidate_group_indices=exact_signature_to_group.get(genotype_signature, ()),
                representative_member_indices=tie_group_member_indices,
                tie_column=tie_column,
                missing_mask=column_missing_mask,
                sign=1.0,
                use_rank_canonicalization=use_rank_canonicalization,
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
                tie_column=tie_column,
                missing_mask=column_missing_mask,
                sign=-1.0,
                use_rank_canonicalization=use_rank_canonicalization,
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
    if len(kept_variant_indices) == n_total and np.array_equal(
        original_to_reduced,
        np.arange(n_total, dtype=np.int32),
    ):
        return _compact_identity_tie_map(n_total)
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


def _build_tie_map_from_hardcall_int8(
    standardized_genotypes: StandardizedGenotypeMatrix,
) -> TieMap:
    n_total = standardized_genotypes.shape[1]
    exact_signature_to_group: dict[tuple[int, int], int | list[int]] = {}
    sign_flipped_signature_to_group: dict[tuple[int, int], int | list[int]] = {}
    tie_group_member_indices: list[list[int]] = []
    tie_group_signs: list[list[float]] = []
    representative_indices: list[int] = []
    kept_variant_indices: list[int] = []
    original_to_reduced = np.full(n_total, -1, dtype=np.int32)
    raw_int8 = cast(Int8BatchCapable, standardized_genotypes.raw)
    variants_done = 0
    local_start = 0
    batch_size = auto_batch_size_i8(standardized_genotypes.shape[0])
    signature_batch_size = _hardcall_tie_signature_batch_size(
        sample_count=standardized_genotypes.shape[0],
        requested_batch_size=batch_size,
    )

    for raw_batch in raw_int8.iter_column_batches_i8(standardized_genotypes.variant_indices, batch_size=batch_size):
        batch_values = np.asarray(raw_batch.values, dtype=np.int8)
        batch_width = int(batch_values.shape[1])
        for signature_start in range(0, batch_width, signature_batch_size):
            signature_stop = min(signature_start + signature_batch_size, batch_width)
            signature_values = batch_values[:, signature_start:signature_stop]
            state_masks = _hardcall_state_masks(signature_values)
            exact_columns, sign_flipped_columns = _canonicalize_hardcall_tie_columns_i8(
                signature_values,
                state_masks,
            )
            exact_signatures = _hashed_tie_signatures_i8(exact_columns)
            sign_flipped_signatures = _hashed_tie_signatures_i8(sign_flipped_columns)

            for local_signature_index, variant_index in enumerate(
                range(local_start + signature_start, local_start + signature_stop)
            ):
                exact_signature = exact_signatures[local_signature_index]
                exact_match_index = _matching_hardcall_tie_group_index(
                    standardized_genotypes=standardized_genotypes,
                    candidate_group_indices=_candidate_group_indices(exact_signature_to_group.get(exact_signature)),
                    representative_member_indices=tie_group_member_indices,
                    tie_column=exact_columns[:, local_signature_index],
                    sign=1.0,
                )
                if exact_match_index is not None:
                    tie_group_member_indices[exact_match_index].append(int(variant_index))
                    tie_group_signs[exact_match_index].append(1.0)
                    original_to_reduced[int(variant_index)] = exact_match_index
                    continue

                sign_flipped_match_index = _matching_hardcall_tie_group_index(
                    standardized_genotypes=standardized_genotypes,
                    candidate_group_indices=_candidate_group_indices(sign_flipped_signature_to_group.get(exact_signature)),
                    representative_member_indices=tie_group_member_indices,
                    tie_column=exact_columns[:, local_signature_index],
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
                _record_candidate_group(exact_signature_to_group, exact_signature, reduced_index)
                _record_candidate_group(sign_flipped_signature_to_group, sign_flipped_signatures[local_signature_index], reduced_index)
                original_to_reduced[int(variant_index)] = reduced_index

        local_start += raw_batch.variant_indices.shape[0]
        variants_done += raw_batch.variant_indices.shape[0]
        if variants_done == raw_batch.variant_indices.shape[0] or variants_done % max(n_total // 10, 1) < raw_batch.variant_indices.shape[0] or variants_done == n_total:
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
    if standardized_genotypes.raw is not None and not _uses_identity_standardization(standardized_genotypes):
        local_start = 0
        raw = standardized_genotypes.raw
        if hasattr(raw, "iter_column_batches_i8"):
            batch_iter = raw.iter_column_batches_i8(
                standardized_genotypes.variant_indices,
                batch_size=batch_size,
            )
            missing_sentinel = PLINK_MISSING_INT8
        else:
            batch_iter = raw.iter_column_batches(
                standardized_genotypes.variant_indices,
                batch_size=batch_size,
            )
            missing_sentinel = None
        for raw_batch in batch_iter:
            local_stop = local_start + raw_batch.variant_indices.shape[0]
            local_indices = np.arange(local_start, local_stop, dtype=np.int32)
            if missing_sentinel is None:
                batch_missing_mask = np.isnan(raw_batch.values)
                batch_values = raw_batch.values
            else:
                batch_missing_mask = raw_batch.values == missing_sentinel
                batch_values = raw_batch.values
            yield RawGenotypeBatch(
                variant_indices=local_indices,
                values=batch_values,
            ), batch_missing_mask
            local_start = local_stop
        return
    for batch in standardized_genotypes.iter_column_batches(batch_size=batch_size):
        yield batch, np.zeros_like(batch.values, dtype=bool)


def _uses_identity_standardization(standardized_genotypes: StandardizedGenotypeMatrix) -> bool:
    return bool(
        np.all(standardized_genotypes.means == 0.0)
        and np.all(standardized_genotypes.scales == 1.0)
    )


def _normalize_signed_zeros(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.where(array == 0.0, np.float32(0.0), array).astype(np.float32, copy=False)


def _standardize_tie_column_from_raw(
    standardized_genotypes: StandardizedGenotypeMatrix,
    raw_column: np.ndarray,
    local_variant_index: int,
) -> np.ndarray:
    raw_variant_index = int(standardized_genotypes.variant_indices[local_variant_index])
    standardized_column = _standardize_batch(
        np.asarray(raw_column, dtype=np.float32).reshape(-1, 1),
        standardized_genotypes.means[np.asarray([raw_variant_index], dtype=np.int32)],
        standardized_genotypes.scales[np.asarray([raw_variant_index], dtype=np.int32)],
    )[:, 0]
    return _normalize_signed_zeros(standardized_column)


def _canonicalize_tie_column(
    column: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(column)
    mask = np.asarray(missing_mask, dtype=bool)

    if values.dtype == np.int8:
        canonical = np.full(values.shape[0], -1, dtype=np.int32)
        if not np.any(~mask):
            return canonical
        observed = values[~mask]
        unique_values = np.unique(observed)
        canonical[~mask] = np.searchsorted(unique_values, observed).astype(np.int32, copy=False)
        return canonical

    values_f32 = _normalize_signed_zeros(np.asarray(values, dtype=np.float32))
    canonical_f32 = np.full(values_f32.shape[0], np.float32(np.nan), dtype=np.float32)
    canonical_f32[~mask] = values_f32[~mask]
    return canonical_f32


def _should_use_rank_canonicalization(
    column: np.ndarray,
    missing_mask: np.ndarray,
) -> bool:
    values = np.asarray(column)
    mask = np.asarray(missing_mask, dtype=bool)
    observed = values[~mask]
    if observed.size == 0:
        return True
    if values.dtype == np.int8:
        return np.unique(observed).shape[0] <= 3
    observed_f32 = _normalize_signed_zeros(np.asarray(observed, dtype=np.float32))
    rounded = np.rint(observed_f32)
    if not np.array_equal(observed_f32, rounded.astype(np.float32, copy=False)):
        return False
    return np.unique(rounded.astype(np.int32, copy=False)).shape[0] <= 3


def _sign_flip_canonical_tie_column(canonical_column: np.ndarray) -> np.ndarray:
    canonical = np.asarray(canonical_column)
    if np.issubdtype(canonical.dtype, np.integer):
        flipped = canonical.copy()
        observed = canonical >= 0
        if not np.any(observed):
            return flipped
        highest_rank = int(np.max(canonical[observed]))
        flipped[observed] = highest_rank - canonical[observed]
        return flipped
    flipped = np.asarray(canonical, dtype=np.float32).copy()
    observed = np.isfinite(flipped)
    flipped[observed] = _normalize_signed_zeros(-flipped[observed])
    return flipped


def _hardcall_state_masks(batch_values: np.ndarray) -> np.ndarray:
    values = np.asarray(batch_values, dtype=np.int8)
    observed = values != PLINK_MISSING_INT8
    return (
        np.any(observed & (values == 0), axis=0).astype(np.uint8)
        + (2 * np.any(observed & (values == 1), axis=0).astype(np.uint8))
        + (4 * np.any(observed & (values == 2), axis=0).astype(np.uint8))
    )


def _hardcall_tie_signature_batch_size(
    sample_count: int,
    requested_batch_size: int,
) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive.")
    bytes_per_variant = sample_count * 2
    memory_capped_batch_size = max(HARD_CALL_TIE_SIGNATURE_TARGET_BYTES // max(bytes_per_variant, 1), 1)
    return max(1, min(int(memory_capped_batch_size), int(requested_batch_size)))


def _canonicalize_hardcall_tie_columns_i8(
    batch_values: np.ndarray,
    state_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(batch_values, dtype=np.int8)
    masks = np.asarray(state_masks, dtype=np.uint8).reshape(-1)
    if values.ndim != 2:
        raise ValueError("batch_values must be a 2D array.")
    if masks.shape[0] != values.shape[1]:
        raise ValueError("state_masks must align with batch columns.")

    encoded_values = np.where(values == PLINK_MISSING_INT8, np.int8(3), values).astype(np.int8, copy=False)
    transposed_codes = encoded_values.T[:, :, None]
    exact_lut = _HARD_CALL_EXACT_SIGNATURE_LUT[masks][:, None, :]
    sign_flipped_lut = _HARD_CALL_SIGN_FLIPPED_SIGNATURE_LUT[masks][:, None, :]
    canonical = np.take_along_axis(exact_lut, transposed_codes, axis=2)[:, :, 0].T
    sign_flipped = np.take_along_axis(sign_flipped_lut, transposed_codes, axis=2)[:, :, 0].T
    return canonical.astype(np.int8, copy=False), sign_flipped.astype(np.int8, copy=False)


def _pack_hardcall_tie_columns_i8(canonical_columns: np.ndarray) -> np.ndarray:
    columns = np.asarray(canonical_columns, dtype=np.uint8)
    if columns.ndim != 2:
        raise ValueError("canonical_columns must be 2D.")
    row_padding = (-columns.shape[0]) % 4
    if row_padding:
        columns = np.pad(columns, ((0, row_padding), (0, 0)), mode="constant", constant_values=0)
    reshaped = columns.reshape(columns.shape[0] // 4, 4, columns.shape[1])
    packed = (
        reshaped[:, 0, :]
        | (reshaped[:, 1, :] << 2)
        | (reshaped[:, 2, :] << 4)
        | (reshaped[:, 3, :] << 6)
    )
    return np.ascontiguousarray(packed.T)


@lru_cache(maxsize=None)
def _hardcall_signature_hash_parameters(word_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if word_count < 1:
        raise ValueError("word_count must be positive.")
    index = np.arange(word_count, dtype=np.uint64) + np.uint64(1)
    multiplier_one = index * np.uint64(0x9E3779B185EBCA87) + np.uint64(0xD6E8FEB86659FD93)
    multiplier_two = index * np.uint64(0x94D049BB133111EB) + np.uint64(0xBF58476D1CE4E5B9)
    mix_multiplier = multiplier_one ^ np.uint64(0x27D4EB2F165667C5)
    return multiplier_one, multiplier_two, mix_multiplier


def _hashed_tie_signatures_i8(canonical_columns: np.ndarray) -> list[tuple[int, int]]:
    packed_columns = _pack_hardcall_tie_columns_i8(canonical_columns)
    byte_padding = (-packed_columns.shape[1]) % np.dtype(np.uint64).itemsize
    if byte_padding:
        packed_columns = np.pad(packed_columns, ((0, 0), (0, byte_padding)), mode="constant", constant_values=0)
    words = packed_columns.view(np.uint64)
    multiplier_one, multiplier_two, mix_multiplier = _hardcall_signature_hash_parameters(words.shape[1])
    rotated_words = (words << np.uint64(17)) | (words >> np.uint64(47))
    signature_one = np.bitwise_xor.reduce(words * multiplier_one[None, :], axis=1)
    signature_two = np.bitwise_xor.reduce((rotated_words + multiplier_two[None, :]) * mix_multiplier[None, :], axis=1)
    return list(zip(signature_one.tolist(), signature_two.tolist(), strict=True))


def _hashed_tie_signature(
    standardized_column: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[bytes, bytes]:
    value_hasher = hashlib.blake2b(digest_size=8, person=b"svpgs_value")
    value_hasher.update(np.ascontiguousarray(standardized_column).view(np.uint8))
    mask_hasher = hashlib.blake2b(digest_size=8, person=b"svpgs_mask_")
    mask_hasher.update(np.packbits(np.ascontiguousarray(missing_mask, dtype=np.uint8)))
    return value_hasher.digest(), mask_hasher.digest()


def _candidate_group_indices(candidate_entry: int | list[int] | None) -> tuple[int, ...]:
    if candidate_entry is None:
        return ()
    if isinstance(candidate_entry, list):
        return tuple(candidate_entry)
    return (int(candidate_entry),)


_CandidateKey = TypeVar("_CandidateKey", bound=Hashable)


def _record_candidate_group(
    candidate_map: dict[_CandidateKey, int | list[int]],
    signature: _CandidateKey,
    reduced_index: int,
) -> None:
    existing = candidate_map.get(signature)
    if existing is None:
        candidate_map[signature] = reduced_index
        return
    if isinstance(existing, list):
        existing.append(reduced_index)
        return
    candidate_map[signature] = [existing, reduced_index]


def _matching_hardcall_tie_group_index(
    standardized_genotypes: StandardizedGenotypeMatrix,
    candidate_group_indices: Sequence[int],
    representative_member_indices: Sequence[list[int]],
    tie_column: np.ndarray,
    sign: float,
) -> int | None:
    for reduced_index in candidate_group_indices:
        representative_column, representative_sign_flipped_column = _load_hardcall_tie_columns(
            standardized_genotypes,
            int(representative_member_indices[int(reduced_index)][0]),
        )
        comparison_column = representative_sign_flipped_column if sign < 0.0 else representative_column
        if np.array_equal(tie_column, comparison_column):
            return int(reduced_index)
    return None


def _matching_tie_group_index(
    standardized_genotypes: StandardizedGenotypeMatrix,
    candidate_group_indices: Sequence[int],
    representative_member_indices: Sequence[list[int]],
    tie_column: np.ndarray,
    missing_mask: np.ndarray,
    sign: float,
    use_rank_canonicalization: bool,
) -> int | None:
    for reduced_index in candidate_group_indices:
        representative_column, representative_missing_mask = _load_tie_column_with_missingness(
            standardized_genotypes,
            int(representative_member_indices[int(reduced_index)][0]),
            use_rank_canonicalization=use_rank_canonicalization,
        )
        if not np.array_equal(missing_mask, representative_missing_mask):
            continue
        if sign < 0.0:
            if use_rank_canonicalization:
                representative_column = _sign_flip_canonical_tie_column(representative_column)
            else:
                representative_column = _normalize_signed_zeros(-representative_column)
        if np.array_equal(tie_column, representative_column):
            return int(reduced_index)
    return None


def _load_hardcall_tie_columns(
    standardized_genotypes: StandardizedGenotypeMatrix,
    local_variant_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if standardized_genotypes.raw is None or not _supports_int8_batches(standardized_genotypes.raw):
        raise RuntimeError("hardcall tie loading requires int8 raw backing storage.")
    raw_int8 = cast(Int8BatchCapable, standardized_genotypes.raw)
    raw_variant_index = int(standardized_genotypes.variant_indices[local_variant_index])
    raw_batch = next(raw_int8.iter_column_batches_i8([raw_variant_index], batch_size=1))
    exact_column, sign_flipped_column = _canonicalize_hardcall_tie_columns_i8(
        np.asarray(raw_batch.values, dtype=np.int8),
        _hardcall_state_masks(raw_batch.values),
    )
    return exact_column[:, 0], sign_flipped_column[:, 0]


def _load_tie_column_with_missingness(
    standardized_genotypes: StandardizedGenotypeMatrix,
    local_variant_index: int,
    *,
    use_rank_canonicalization: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if standardized_genotypes.raw is not None:
        raw_variant_index = int(standardized_genotypes.variant_indices[local_variant_index])
        raw_column = standardized_genotypes.raw.materialize([raw_variant_index]).reshape(-1)
        missing_mask = np.isnan(raw_column)
        if use_rank_canonicalization:
            rank_source = np.rint(np.asarray(raw_column, dtype=np.float32)).astype(np.int8, copy=False)
            return _canonicalize_tie_column(rank_source, missing_mask), missing_mask
        return _standardize_tie_column_from_raw(standardized_genotypes, raw_column, local_variant_index), missing_mask
    if standardized_genotypes._dense_cache is None:
        dense_column = standardized_genotypes.materialize()[:, local_variant_index]
    else:
        dense_column = standardized_genotypes._dense_cache[:, local_variant_index]
    if use_rank_canonicalization:
        missing_mask = np.zeros(dense_column.shape[0], dtype=bool)
        rank_source = np.rint(np.asarray(dense_column, dtype=np.float32)).astype(np.int8, copy=False)
        return _canonicalize_tie_column(rank_source, missing_mask), missing_mask
    return _normalize_signed_zeros(dense_column), np.zeros(dense_column.shape[0], dtype=bool)


def collapse_tie_groups(
    records: Sequence[VariantRecord],
    tie_map: TieMap,
) -> list[VariantRecord]:
    """Create one merged record per tie group for use in the reduced model."""
    if not tie_map.reduced_to_group:
        kept_indices = np.asarray(tie_map.kept_indices, dtype=np.int32)
        if kept_indices.shape[0] == len(records) and np.array_equal(kept_indices, np.arange(len(records), dtype=np.int32)):
            return records if isinstance(records, list) else list(records)
        return [records[int(record_index)] for record_index in kept_indices]
    if len(tie_map.reduced_to_group) == len(records):
        record_indices = np.arange(len(records), dtype=np.int32)
        if (
            np.array_equal(tie_map.kept_indices, record_indices)
            and np.array_equal(tie_map.original_to_reduced, record_indices)
        ):
            return records if isinstance(records, list) else list(records)

    collapsed_records: list[VariantRecord] = []
    for tie_group in tie_map.reduced_to_group:
        member_indices = tie_group.member_indices
        if member_indices.shape[0] == 1:
            collapsed_records.append(records[int(member_indices[0])])
            continue
        member_records = [records[int(member_index)] for member_index in member_indices]
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
        binary_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_binary_features
            }
        )
        categorical_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_categorical_features
            }
        )
        nested_feature_names = sorted(
            {
                feature_name
                for member_record in member_records
                for feature_name in member_record.prior_nested_features
            }
        )
        collapsed_binary_features: dict[str, bool] = {}
        collapsed_categorical_features: dict[str, str] = {}
        collapsed_membership_features = _average_weighted_feature_dicts(
            [
                member_record.prior_membership_features
                for member_record in member_records
            ]
        )
        collapsed_nested_features: dict[str, tuple[str, ...]] = {}
        collapsed_nested_membership_features = _average_weighted_feature_dicts(
            [
                member_record.prior_nested_membership_features
                for member_record in member_records
            ]
        )

        for feature_name in binary_feature_names:
            true_frequency = float(
                np.mean(
                    [
                        float(member_record.prior_binary_features.get(feature_name, False))
                        for member_record in member_records
                    ]
                )
            )
            if np.isclose(true_frequency, 0.0):
                collapsed_binary_features[feature_name] = False
                continue
            if np.isclose(true_frequency, 1.0):
                collapsed_binary_features[feature_name] = True
                continue
            collapsed_membership_features[feature_name] = {
                "false": 1.0 - true_frequency,
                "true": true_frequency,
            }

        for feature_name in categorical_feature_names:
            feature_values = [
                member_record.prior_categorical_features.get(feature_name)
                for member_record in member_records
            ]
            observed_feature_values = [feature_value for feature_value in feature_values if feature_value is not None]
            if not observed_feature_values:
                continue
            if len(set(observed_feature_values)) == 1 and len(observed_feature_values) == len(member_records):
                collapsed_categorical_features[feature_name] = observed_feature_values[0]
                continue
            collapsed_membership_features[feature_name] = {
                feature_value: float(np.mean([value == feature_value for value in feature_values]))
                for feature_value in sorted(set(observed_feature_values))
            }

        for feature_name in nested_feature_names:
            feature_paths = [
                member_record.prior_nested_features.get(feature_name)
                for member_record in member_records
            ]
            observed_feature_paths = [feature_path for feature_path in feature_paths if feature_path is not None]
            if not observed_feature_paths:
                continue
            if len(set(observed_feature_paths)) == 1 and len(observed_feature_paths) == len(member_records):
                collapsed_nested_features[feature_name] = observed_feature_paths[0]
                continue
            collapsed_nested_membership_features[feature_name] = {
                ">".join(feature_path): float(np.mean([path_value == feature_path for path_value in feature_paths]))
                for feature_path in sorted(set(observed_feature_paths))
            }

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
                prior_binary_features=collapsed_binary_features,
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
                prior_categorical_features=collapsed_categorical_features,
                prior_membership_features=collapsed_membership_features,
                prior_nested_features=collapsed_nested_features,
                prior_nested_membership_features=collapsed_nested_membership_features,
                prior_class_members=tuple(unique_variant_classes),
                prior_class_membership=tuple(class_membership.tolist()),
            )
        )
    return collapsed_records


def _average_weighted_feature_dicts(
    weighted_feature_dicts: Sequence[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    averaged_features: dict[str, dict[str, float]] = {}
    feature_names = sorted(
        {
            feature_name
            for feature_dict in weighted_feature_dicts
            for feature_name in feature_dict
        }
    )
    for feature_name in feature_names:
        level_names = sorted(
            {
                level_name
                for feature_dict in weighted_feature_dicts
                for level_name in feature_dict.get(feature_name, {})
            }
        )
        averaged_levels = {
            level_name: float(
                np.mean(
                    [
                        feature_dict.get(feature_name, {}).get(level_name, 0.0)
                        for feature_dict in weighted_feature_dicts
                    ]
                )
            )
            for level_name in level_names
        }
        nonzero_levels = {
            level_name: level_weight
            for level_name, level_weight in averaged_levels.items()
            if level_weight > 0.0
        }
        if nonzero_levels:
            averaged_features[feature_name] = nonzero_levels
    return averaged_features


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

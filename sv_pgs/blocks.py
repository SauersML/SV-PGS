"""LD block construction, low-rank preconditioning, and variance estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from sv_pgs.config import ModelConfig
from sv_pgs.data import VariantRecord


@dataclass(frozen=True, slots=True)
class LDBlock:
    variant_indices: jnp.ndarray
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray
    condition_number: float
    jitter: float


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class BlockDecomposition:
    blocks: tuple[LDBlock, ...]
    variant_to_block: jnp.ndarray
    block_variant_indices: jnp.ndarray
    block_variant_mask: jnp.ndarray
    block_eigenvalues: jnp.ndarray
    block_eigenvectors: jnp.ndarray
    block_jitter: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.variant_to_block,
            self.block_variant_indices,
            self.block_variant_mask,
            self.block_eigenvalues,
            self.block_eigenvectors,
            self.block_jitter,
        )
        auxiliary_data = None
        return children, auxiliary_data

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        return cls(
            blocks=(),
            variant_to_block=children[0],
            block_variant_indices=children[1],
            block_variant_mask=children[2],
            block_eigenvalues=children[3],
            block_eigenvectors=children[4],
            block_jitter=children[5],
        )


def build_block_decomposition(
    genotypes: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
    sample_weights: np.ndarray | None = None,
    device: jax.Device | None = None,
) -> BlockDecomposition:
    variant_count = genotypes.shape[1]
    sample_count = genotypes.shape[0]
    chromosome_groups = _group_by_chromosome(records)
    all_blocks: list[LDBlock] = []
    variant_to_block = np.full(variant_count, -1, dtype=np.int32)

    for chromosome_indices in chromosome_groups:
        sorted_indices = _sort_by_position(chromosome_indices, records)
        partitions = _partition_chromosome(
            sorted_indices,
            records,
            config.ld_block_max_variants,
            config.ld_block_window_bp,
        )
        for partition_indices in partitions:
            ld_block = _eigendecompose_block(
                genotypes[:, partition_indices],
                partition_indices,
                sample_count,
                config,
                sample_weights=sample_weights,
            )
            variant_to_block[partition_indices] = len(all_blocks)
            all_blocks.append(ld_block)

    (
        block_variant_indices,
        block_variant_mask,
        block_eigenvalues,
        block_eigenvectors,
        block_jitter,
    ) = _pack_block_arrays(tuple(all_blocks))
    if device is not None:
        block_variant_indices = jax.device_put(block_variant_indices, device=device)
        block_variant_mask = jax.device_put(block_variant_mask, device=device)
        block_eigenvalues = jax.device_put(block_eigenvalues, device=device)
        block_eigenvectors = jax.device_put(block_eigenvectors, device=device)
        block_jitter = jax.device_put(block_jitter, device=device)

    return BlockDecomposition(
        blocks=tuple(all_blocks),
        variant_to_block=jax.device_put(jnp.asarray(variant_to_block, dtype=jnp.int32), device=device),
        block_variant_indices=block_variant_indices,
        block_variant_mask=block_variant_mask,
        block_eigenvalues=block_eigenvalues,
        block_eigenvectors=block_eigenvectors,
        block_jitter=block_jitter,
    )


def refresh_block_decomposition(
    block_decomposition: BlockDecomposition,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    config: ModelConfig,
    device: jax.Device | None = None,
) -> BlockDecomposition:
    if not block_decomposition.blocks:
        return block_decomposition

    sample_count = genotypes.shape[0]
    refreshed_blocks: list[LDBlock] = []
    for block in block_decomposition.blocks:
        block_indices = np.asarray(block.variant_indices, dtype=np.int32)
        refreshed_blocks.append(
            _eigendecompose_block(
                genotypes[:, block_indices],
                block_indices,
                sample_count,
                config,
                sample_weights=sample_weights,
            )
        )

    (
        block_variant_indices,
        block_variant_mask,
        block_eigenvalues,
        block_eigenvectors,
        block_jitter,
    ) = _pack_block_arrays(tuple(refreshed_blocks))
    if device is not None:
        block_variant_indices = jax.device_put(block_variant_indices, device=device)
        block_variant_mask = jax.device_put(block_variant_mask, device=device)
        block_eigenvalues = jax.device_put(block_eigenvalues, device=device)
        block_eigenvectors = jax.device_put(block_eigenvectors, device=device)
        block_jitter = jax.device_put(block_jitter, device=device)

    return BlockDecomposition(
        blocks=tuple(refreshed_blocks),
        variant_to_block=jax.device_put(block_decomposition.variant_to_block, device=device),
        block_variant_indices=block_variant_indices,
        block_variant_mask=block_variant_mask,
        block_eigenvalues=block_eigenvalues,
        block_eigenvectors=block_eigenvectors,
        block_jitter=block_jitter,
    )


def compute_block_posterior(
    block: LDBlock,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    residual_vector: np.ndarray,
    prior_precision_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block_indices = np.asarray(block.variant_indices, dtype=np.int32)
    block_genotype_matrix = jnp.asarray(genotypes[:, block_indices], dtype=jnp.float32)
    block_prior_precision = jnp.asarray(prior_precision_diagonal[block_indices], dtype=jnp.float32)
    sample_weight_array = jnp.asarray(sample_weights, dtype=jnp.float32)
    residual_array = jnp.asarray(residual_vector, dtype=jnp.float32)
    block_size = block_indices.shape[0]

    if block_size == 1:
        weighted_inner_product = float(
            jnp.sum(sample_weight_array * block_genotype_matrix[:, 0] * block_genotype_matrix[:, 0])
        )
        posterior_precision = weighted_inner_product + float(block_prior_precision[0]) + block.jitter
        posterior_variance = np.array([1.0 / max(posterior_precision, 1e-12)], dtype=np.float32)
        posterior_mean = posterior_variance * float(
            jnp.sum(block_genotype_matrix[:, 0] * residual_array)
        )
        return posterior_mean.astype(np.float32), posterior_variance

    eigenvector_basis = block.eigenvectors
    rank = block.eigenvalues.shape[0]
    weighted_right_hand_side = block_genotype_matrix.T @ residual_array
    projected_right_hand_side = eigenvector_basis.T @ weighted_right_hand_side
    weighted_block_genotypes = sample_weight_array[:, None] * block_genotype_matrix
    weighted_gram_matrix_in_eigen_space = (
        eigenvector_basis.T
        @ (block_genotype_matrix.T @ weighted_block_genotypes)
        @ eigenvector_basis
    )
    prior_precision_in_eigen_space = (
        eigenvector_basis.T
        @ (block_prior_precision[:, None] * eigenvector_basis)
    )
    posterior_precision_eigen = (
        weighted_gram_matrix_in_eigen_space
        + prior_precision_in_eigen_space
        + jnp.eye(rank, dtype=jnp.float32) * jnp.float32(block.jitter)
    )
    cholesky_factor = jnp.linalg.cholesky(posterior_precision_eigen)
    eigen_mean = jnp.linalg.solve(
        cholesky_factor.T,
        jnp.linalg.solve(cholesky_factor, projected_right_hand_side),
    )
    posterior_covariance_eigen = jnp.linalg.solve(
        cholesky_factor.T,
        jnp.linalg.solve(cholesky_factor, jnp.eye(rank, dtype=jnp.float32)),
    )
    block_mean = eigenvector_basis @ eigen_mean
    block_variance_diagonal = jnp.sum(
        eigenvector_basis * (eigenvector_basis @ posterior_covariance_eigen),
        axis=1,
    )
    return (
        np.asarray(block_mean, dtype=np.float32),
        np.asarray(jnp.maximum(block_variance_diagonal, 1e-12), dtype=np.float32),
    )


@jax.jit
def apply_block_preconditioner(
    block_decomposition: BlockDecomposition,
    vector: jnp.ndarray,
    prior_precision: jnp.ndarray,
    sample_weight_sum: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the low-rank block preconditioner M^{-1} v."""

    def body(result: jnp.ndarray, scan_inputs: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, None]:
        block_indices, block_mask, block_eigenvalues, block_eigenvectors, _block_jitter = scan_inputs
        del _block_jitter
        block_mask = block_mask.astype(jnp.float32)
        block_vector = vector[block_indices] * block_mask
        block_prior_precision = jnp.maximum(prior_precision[block_indices], 1e-12)
        inverse_sqrt_prior = jnp.reciprocal(jnp.sqrt(block_prior_precision))
        scaled_vector = inverse_sqrt_prior * block_vector * block_mask

        scaled_eigenvalues = block_eigenvalues * sample_weight_sum
        correction_weights = jnp.where(
            block_eigenvalues > 0.0,
            scaled_eigenvalues / (1.0 + scaled_eigenvalues),
            0.0,
        )
        projected = block_eigenvectors.T @ scaled_vector
        corrected = scaled_vector - block_eigenvectors @ (correction_weights * projected)
        block_result = inverse_sqrt_prior * corrected * block_mask
        return result.at[block_indices].add(block_result), None

    initial = jnp.zeros_like(vector, dtype=jnp.float32)
    result, _ = lax.scan(
        body,
        initial,
        (
            block_decomposition.block_variant_indices,
            block_decomposition.block_variant_mask,
            block_decomposition.block_eigenvalues,
            block_decomposition.block_eigenvectors,
            block_decomposition.block_jitter,
        ),
    )
    return result


@jax.jit
def estimate_variance_from_blocks(
    block_decomposition: BlockDecomposition,
    prior_precision: jnp.ndarray,
    sample_weight_sum: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate diag((X^T W X + D)^{-1}) from the block low-rank inverse."""

    def body(variance_diagonal: jnp.ndarray, scan_inputs: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, None]:
        block_indices, block_mask, block_eigenvalues, block_eigenvectors, _block_jitter = scan_inputs
        del _block_jitter
        block_mask = block_mask.astype(jnp.float32)
        block_prior_precision = jnp.maximum(prior_precision[block_indices], 1e-12)
        inverse_prior_precision = jnp.reciprocal(block_prior_precision)
        inverse_sqrt_prior = jnp.sqrt(inverse_prior_precision)

        scaled_eigenvalues = block_eigenvalues * sample_weight_sum
        shrinkage_weights = jnp.where(
            block_eigenvalues > 0.0,
            scaled_eigenvalues / (1.0 + scaled_eigenvalues),
            0.0,
        )
        scaled_eigenvectors = block_eigenvectors * inverse_sqrt_prior[:, None]
        correction_diagonal = jnp.sum(
            scaled_eigenvectors * (scaled_eigenvectors * shrinkage_weights[None, :]),
            axis=1,
        )
        block_variance = (inverse_prior_precision - correction_diagonal) * block_mask
        return variance_diagonal.at[block_indices].add(block_variance), None

    initial = jnp.zeros(prior_precision.shape[0], dtype=jnp.float32)
    variance_diagonal, _ = lax.scan(
        body,
        initial,
        (
            block_decomposition.block_variant_indices,
            block_decomposition.block_variant_mask,
            block_decomposition.block_eigenvalues,
            block_decomposition.block_eigenvectors,
            block_decomposition.block_jitter,
        ),
    )
    return jnp.maximum(variance_diagonal, 1e-12)


def _group_by_chromosome(records: Sequence[VariantRecord]) -> list[np.ndarray]:
    chromosome_to_indices: dict[str, list[int]] = {}
    for variant_index, record in enumerate(records):
        chromosome_to_indices.setdefault(record.chromosome, []).append(variant_index)
    return [np.asarray(indices, dtype=np.int32) for indices in chromosome_to_indices.values()]


def _sort_by_position(indices: np.ndarray, records: Sequence[VariantRecord]) -> np.ndarray:
    positions = np.array([records[index].position for index in indices], dtype=np.int64)
    return indices[np.argsort(positions)]


def _partition_chromosome(
    sorted_indices: np.ndarray,
    records: Sequence[VariantRecord],
    max_block_size: int,
    window_bp: int,
) -> list[np.ndarray]:
    partitions: list[np.ndarray] = []
    current_start = 0
    total_count = sorted_indices.shape[0]
    while current_start < total_count:
        current_end = min(current_start + max_block_size, total_count)
        start_position = records[int(sorted_indices[current_start])].position
        while current_end < total_count:
            next_position = records[int(sorted_indices[current_end])].position
            if next_position - start_position > window_bp:
                break
            if current_end - current_start >= max_block_size:
                break
            current_end += 1
        partitions.append(sorted_indices[current_start:current_end])
        current_start = current_end
    return partitions


def _eigendecompose_block(
    block_genotypes: np.ndarray,
    block_indices: np.ndarray,
    sample_count: int,
    config: ModelConfig,
    sample_weights: np.ndarray | None = None,
) -> LDBlock:
    block_size = block_genotypes.shape[1]
    if block_size == 1:
        single_variant_values = np.asarray(block_genotypes[:, 0], dtype=np.float32)
        if sample_weights is None:
            normalization = max(float(sample_count), 1e-12)
            weighted_second_moment = float(np.dot(single_variant_values, single_variant_values) / normalization)
        else:
            sample_weight_array = np.asarray(sample_weights, dtype=np.float32)
            normalization = max(float(np.sum(sample_weight_array)), 1e-12)
            weighted_second_moment = float(
                np.dot(sample_weight_array * single_variant_values, single_variant_values) / normalization
            )
        return LDBlock(
            variant_indices=jnp.asarray(block_indices, dtype=jnp.int32),
            eigenvalues=jnp.asarray([max(weighted_second_moment, 1e-12)], dtype=jnp.float32),
            eigenvectors=jnp.ones((1, 1), dtype=jnp.float32),
            condition_number=1.0,
            jitter=config.block_jitter_floor,
        )

    block_matrix = jnp.asarray(block_genotypes, dtype=jnp.float32)
    if sample_weights is None:
        weighted_block_matrix = block_matrix
        normalization = float(sample_count)
    else:
        sample_weight_array = jnp.asarray(sample_weights, dtype=jnp.float32)
        sqrt_sample_weights = jnp.sqrt(jnp.maximum(sample_weight_array, 0.0))
        weighted_block_matrix = block_matrix * sqrt_sample_weights[:, None]
        normalization = max(float(np.sum(sample_weights)), 1e-12)

    correlation_matrix = (weighted_block_matrix.T @ weighted_block_matrix) / normalization
    correlation_matrix = correlation_matrix + (
        jnp.eye(block_size, dtype=jnp.float32) * jnp.float32(config.block_jitter_floor)
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(correlation_matrix)
    descending_order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(eigenvalues[descending_order], 1e-12)
    eigenvectors = eigenvectors[:, descending_order]

    total_variance = float(jnp.sum(eigenvalues))
    omitted_variance = total_variance - np.cumsum(np.asarray(eigenvalues, dtype=np.float32))
    maximum_omitted_variance = max(total_variance * config.discarded_spectrum_tolerance, 1e-12)
    retained_count = block_size
    for retained_index, remaining_variance in enumerate(omitted_variance, start=1):
        if remaining_variance <= maximum_omitted_variance:
            retained_count = retained_index
            break

    retained_eigenvalues = eigenvalues[:retained_count]
    retained_eigenvectors = eigenvectors[:, :retained_count]
    condition_number = float(retained_eigenvalues[0] / max(float(retained_eigenvalues[-1]), 1e-12))
    block_jitter = max(
        config.block_jitter_floor,
        config.block_jitter_floor * condition_number * 1e-6,
    )
    return LDBlock(
        variant_indices=jnp.asarray(block_indices, dtype=jnp.int32),
        eigenvalues=retained_eigenvalues.astype(jnp.float32),
        eigenvectors=retained_eigenvectors.astype(jnp.float32),
        condition_number=condition_number,
        jitter=block_jitter,
    )


def _pack_block_arrays(
    blocks: tuple[LDBlock, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if not blocks:
        empty_int = jnp.zeros((0, 0), dtype=jnp.int32)
        empty_float = jnp.zeros((0, 0), dtype=jnp.float32)
        empty_eigenvectors = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        return empty_int, empty_float, empty_float, empty_eigenvectors, jnp.zeros(0, dtype=jnp.float32)

    max_block_size = max(int(block.variant_indices.shape[0]) for block in blocks)
    max_rank = max(int(block.eigenvalues.shape[0]) for block in blocks)
    block_count = len(blocks)

    block_variant_indices = np.zeros((block_count, max_block_size), dtype=np.int32)
    block_variant_mask = np.zeros((block_count, max_block_size), dtype=np.float32)
    block_eigenvalues = np.zeros((block_count, max_rank), dtype=np.float32)
    block_eigenvectors = np.zeros((block_count, max_block_size, max_rank), dtype=np.float32)
    block_jitter = np.zeros(block_count, dtype=np.float32)

    for block_index, block in enumerate(blocks):
        block_size = int(block.variant_indices.shape[0])
        rank = int(block.eigenvalues.shape[0])
        block_variant_indices[block_index, :block_size] = np.asarray(block.variant_indices, dtype=np.int32)
        block_variant_mask[block_index, :block_size] = 1.0
        block_eigenvalues[block_index, :rank] = np.asarray(block.eigenvalues, dtype=np.float32)
        block_eigenvectors[block_index, :block_size, :rank] = np.asarray(block.eigenvectors, dtype=np.float32)
        block_jitter[block_index] = np.float32(block.jitter)

    return (
        jnp.asarray(block_variant_indices, dtype=jnp.int32),
        jnp.asarray(block_variant_mask, dtype=jnp.float32),
        jnp.asarray(block_eigenvalues, dtype=jnp.float32),
        jnp.asarray(block_eigenvectors, dtype=jnp.float32),
        jnp.asarray(block_jitter, dtype=jnp.float32),
    )

"""LD block construction, eigendecomposition, and blockwise posterior updates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs.config import ModelConfig
from sv_pgs.data import VariantRecord


@dataclass(slots=True)
class LDBlock:
    variant_indices: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    jitter: float


@dataclass(slots=True)
class BlockDecomposition:
    blocks: list[LDBlock]
    variant_to_block: np.ndarray


def build_block_decomposition(
    genotypes: np.ndarray, records: Sequence[VariantRecord], config: ModelConfig,
) -> BlockDecomposition:
    variant_count = genotypes.shape[1]
    sample_count = genotypes.shape[0]
    chromosome_groups = _group_by_chromosome(records)
    all_blocks: list[LDBlock] = []
    variant_to_block = np.full(variant_count, -1, dtype=np.int32)
    for chromosome_indices in chromosome_groups:
        sorted_indices = _sort_by_position(chromosome_indices, records)
        partitions = _partition_chromosome(sorted_indices, records, config.ld_block_max_variants, config.ld_block_window_bp)
        for partition_indices in partitions:
            block_genotypes = genotypes[:, partition_indices]
            ld_block = _eigendecompose_block(block_genotypes, partition_indices, sample_count, config)
            block_index = len(all_blocks)
            all_blocks.append(ld_block)
            variant_to_block[partition_indices] = block_index
    return BlockDecomposition(blocks=all_blocks, variant_to_block=variant_to_block)


def compute_block_posterior(
    block: LDBlock, genotypes: np.ndarray, sample_weights: np.ndarray,
    residual_vector: np.ndarray, prior_precision_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block_indices = block.variant_indices
    block_genotype_matrix = genotypes[:, block_indices]
    block_prior_precision = prior_precision_diagonal[block_indices]
    block_size = block_indices.shape[0]
    if block_size == 1:
        weighted_inner_product = float(np.sum(sample_weights * block_genotype_matrix[:, 0] * block_genotype_matrix[:, 0]))
        posterior_precision = weighted_inner_product + block_prior_precision[0] + block.jitter
        posterior_variance = np.array([1.0 / max(posterior_precision, 1e-12)], dtype=np.float64)
        posterior_mean = posterior_variance * np.sum(block_genotype_matrix[:, 0] * residual_vector)
        return posterior_mean.astype(np.float32), posterior_variance.astype(np.float32)
    eigenvector_basis = block.eigenvectors
    rank = block.eigenvalues.shape[0]
    weighted_right_hand_side = block_genotype_matrix.T @ residual_vector
    projected_right_hand_side = eigenvector_basis.T @ weighted_right_hand_side
    weighted_block_genotypes = sample_weights[:, None] * block_genotype_matrix
    weighted_gram_matrix_in_eigen_space = (
        eigenvector_basis.T @ (block_genotype_matrix.T @ weighted_block_genotypes) @ eigenvector_basis
    )
    prior_precision_in_eigen_space = eigenvector_basis.T @ np.diag(block_prior_precision) @ eigenvector_basis
    posterior_precision_eigen = weighted_gram_matrix_in_eigen_space + prior_precision_in_eigen_space
    posterior_precision_eigen += np.eye(rank, dtype=np.float64) * block.jitter
    try:
        cholesky_factor = np.linalg.cholesky(posterior_precision_eigen)
        eigen_mean = np.linalg.solve(cholesky_factor.T, np.linalg.solve(cholesky_factor, projected_right_hand_side))
        posterior_cov_eigen = np.linalg.solve(
            cholesky_factor.T,
            np.linalg.solve(cholesky_factor, np.eye(rank, dtype=np.float64)),
        )
    except np.linalg.LinAlgError:
        posterior_precision_eigen += np.eye(rank, dtype=np.float64) * 1e-4
        eigen_mean = np.linalg.solve(posterior_precision_eigen, projected_right_hand_side)
        posterior_cov_eigen = np.linalg.inv(posterior_precision_eigen)
    block_mean = eigenvector_basis @ eigen_mean
    block_variance_diag = np.sum(eigenvector_basis * (eigenvector_basis @ posterior_cov_eigen), axis=1)
    block_variance_diag = np.maximum(block_variance_diag, 1e-12)
    return block_mean.astype(np.float32), block_variance_diag.astype(np.float32)


def _group_by_chromosome(records: Sequence[VariantRecord]) -> list[np.ndarray]:
    chromosome_to_indices: dict[str, list[int]] = {}
    for variant_index, record in enumerate(records):
        chromosome_to_indices.setdefault(record.chromosome, []).append(variant_index)
    return [np.asarray(indices, dtype=np.int32) for indices in chromosome_to_indices.values()]


def _sort_by_position(indices: np.ndarray, records: Sequence[VariantRecord]) -> np.ndarray:
    positions = np.array([records[idx].position for idx in indices], dtype=np.int64)
    return indices[np.argsort(positions)]


def _partition_chromosome(
    sorted_indices: np.ndarray, records: Sequence[VariantRecord], max_block_size: int, window_bp: int,
) -> list[np.ndarray]:
    partitions: list[np.ndarray] = []
    current_start = 0
    total = sorted_indices.shape[0]
    while current_start < total:
        current_end = min(current_start + max_block_size, total)
        start_position = records[sorted_indices[current_start]].position
        while current_end < total:
            next_position = records[sorted_indices[current_end]].position
            if next_position - start_position > window_bp or current_end - current_start >= max_block_size:
                break
            current_end += 1
        partitions.append(sorted_indices[current_start:current_end])
        current_start = current_end
    return partitions


def _eigendecompose_block(
    block_genotypes: np.ndarray, block_indices: np.ndarray, sample_count: int, config: ModelConfig,
) -> LDBlock:
    block_size = block_genotypes.shape[1]
    if block_size == 1:
        return LDBlock(
            variant_indices=block_indices, eigenvalues=np.ones(1, dtype=np.float64),
            eigenvectors=np.ones((1, 1), dtype=np.float64), condition_number=1.0, jitter=config.block_jitter_floor,
        )
    correlation_matrix = (block_genotypes.T @ block_genotypes).astype(np.float64) / sample_count
    correlation_matrix += np.eye(block_size, dtype=np.float64) * config.block_jitter_floor
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[descending_order], 1e-12)
    eigenvectors = eigenvectors[:, descending_order]
    total_variance = float(np.sum(eigenvalues))
    omitted_variance = total_variance - np.cumsum(eigenvalues)
    maximum_omitted_variance = max(total_variance * config.discarded_spectrum_tolerance, 1e-12)
    retained_count = block_size
    for retained_index, remaining_variance in enumerate(omitted_variance, start=1):
        if remaining_variance <= maximum_omitted_variance:
            retained_count = retained_index
            break
    retained_eigenvalues = eigenvalues[:retained_count]
    retained_eigenvectors = eigenvectors[:, :retained_count]
    condition_number = float(retained_eigenvalues[0] / max(retained_eigenvalues[-1], 1e-12))
    block_jitter = max(config.block_jitter_floor, config.block_jitter_floor * condition_number * 1e-6)
    return LDBlock(
        variant_indices=block_indices, eigenvalues=retained_eigenvalues,
        eigenvectors=retained_eigenvectors, condition_number=condition_number, jitter=block_jitter,
    )

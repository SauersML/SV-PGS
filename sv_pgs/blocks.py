"""LD block construction, eigendecomposition, and blockwise posterior updates.

Blocks are built from the joint SNP+SV training genotype matrix.
Hard blocks are eigendecomposed with tolerance-driven truncation.
Posterior updates run in the retained eigenspace for stability and speed.
"""
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
    genotypes: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> BlockDecomposition:
    variant_count = genotypes.shape[1]
    sample_count = genotypes.shape[0]
    chromosome_groups = _group_by_chromosome(records)
    all_blocks: list[LDBlock] = []
    variant_to_block = np.full(variant_count, -1, dtype=np.int32)

    for chromosome_indices in chromosome_groups:
        sorted_indices = _sort_by_position(chromosome_indices, records)
        partitions = _partition_chromosome(
            sorted_indices, records, config.ld_block_max_variants, config.ld_block_window_bp,
        )
        for partition_indices in partitions:
            block_genotypes = genotypes[:, partition_indices]
            ld_block = _eigendecompose_block(
                block_genotypes, partition_indices, sample_count, config,
            )
            block_index = len(all_blocks)
            all_blocks.append(ld_block)
            variant_to_block[partition_indices] = block_index

    return BlockDecomposition(blocks=all_blocks, variant_to_block=variant_to_block)


def compute_block_posterior(
    block: LDBlock,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    residual_vector: np.ndarray,
    prior_precision_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blockwise Gaussian posterior in eigenspace. Returns (mean, variance_diagonal)."""
    block_indices = block.variant_indices
    block_x = genotypes[:, block_indices]
    block_prior_prec = prior_precision_diagonal[block_indices]
    block_size = block_indices.shape[0]

    if block_size == 1:
        xtw_x = float(np.sum(sample_weights * block_x[:, 0] * block_x[:, 0]))
        posterior_prec = xtw_x + block_prior_prec[0] + block.jitter
        posterior_var = np.array([1.0 / max(posterior_prec, 1e-12)], dtype=np.float64)
        posterior_mean = posterior_var * np.sum(block_x[:, 0] * residual_vector)
        return posterior_mean.astype(np.float32), posterior_var.astype(np.float32)

    eigvecs = block.eigenvectors
    eigvals = block.eigenvalues
    rank = eigvals.shape[0]

    xtw_r = block_x.T @ residual_vector
    projected_rhs = eigvecs.T @ xtw_r

    wx = sample_weights[:, None] * block_x
    xtwx_in_eigen = eigvecs.T @ (block_x.T @ wx) @ eigvecs
    prior_prec_in_eigen = eigvecs.T @ np.diag(block_prior_prec) @ eigvecs
    posterior_precision_eigen = xtwx_in_eigen + prior_prec_in_eigen
    posterior_precision_eigen += np.eye(rank, dtype=np.float64) * block.jitter

    try:
        cholesky_factor = np.linalg.cholesky(posterior_precision_eigen)
        eigen_mean = np.linalg.solve(
            cholesky_factor.T,
            np.linalg.solve(cholesky_factor, projected_rhs),
        )
        posterior_cov_eigen = np.linalg.solve(
            cholesky_factor.T,
            np.linalg.solve(cholesky_factor, np.eye(rank, dtype=np.float64)),
        )
    except np.linalg.LinAlgError:
        posterior_precision_eigen += np.eye(rank, dtype=np.float64) * 1e-4
        eigen_mean = np.linalg.solve(posterior_precision_eigen, projected_rhs)
        posterior_cov_eigen = np.linalg.inv(posterior_precision_eigen)

    block_mean = eigvecs @ eigen_mean
    block_variance_diag = np.sum(eigvecs * (eigvecs @ posterior_cov_eigen), axis=1)
    block_variance_diag = np.maximum(block_variance_diag, 1e-12)

    return block_mean.astype(np.float32), block_variance_diag.astype(np.float32)


def _group_by_chromosome(records: Sequence[VariantRecord]) -> list[np.ndarray]:
    chromosome_to_indices: dict[str, list[int]] = {}
    for variant_index, record in enumerate(records):
        chromosome_to_indices.setdefault(record.chromosome, []).append(variant_index)
    return [np.asarray(indices, dtype=np.int32) for indices in chromosome_to_indices.values()]


def _sort_by_position(
    indices: np.ndarray, records: Sequence[VariantRecord],
) -> np.ndarray:
    positions = np.array([records[idx].position for idx in indices], dtype=np.int64)
    sort_order = np.argsort(positions)
    return indices[sort_order]


def _partition_chromosome(
    sorted_indices: np.ndarray,
    records: Sequence[VariantRecord],
    max_block_size: int,
    window_bp: int,
) -> list[np.ndarray]:
    partitions: list[np.ndarray] = []
    current_start = 0
    total = sorted_indices.shape[0]

    while current_start < total:
        current_end = min(current_start + max_block_size, total)
        start_position = records[sorted_indices[current_start]].position
        while current_end < total:
            next_position = records[sorted_indices[current_end]].position
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
) -> LDBlock:
    block_size = block_genotypes.shape[1]

    if block_size == 1:
        return LDBlock(
            variant_indices=block_indices,
            eigenvalues=np.ones(1, dtype=np.float64),
            eigenvectors=np.ones((1, 1), dtype=np.float64),
            condition_number=1.0,
            jitter=config.block_jitter_floor,
        )

    correlation_matrix = (block_genotypes.T @ block_genotypes).astype(np.float64) / sample_count
    correlation_matrix += np.eye(block_size, dtype=np.float64) * config.block_jitter_floor

    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_order]
    eigenvectors = eigenvectors[:, descending_order]

    eigenvalues = np.maximum(eigenvalues, 1e-12)
    total_variance = float(np.sum(eigenvalues))
    cumulative_variance = np.cumsum(eigenvalues)
    retained_threshold = total_variance * (1.0 - config.eigenvalue_tolerance)

    retained_count = int(np.searchsorted(cumulative_variance, retained_threshold) + 1)
    retained_count = min(max(retained_count, 1), block_size)

    retained_eigenvalues = eigenvalues[:retained_count]
    retained_eigenvectors = eigenvectors[:, :retained_count]
    condition_number = float(retained_eigenvalues[0] / max(retained_eigenvalues[-1], 1e-12))
    block_jitter = max(config.block_jitter_floor, config.block_jitter_floor * condition_number * 1e-6)

    return LDBlock(
        variant_indices=block_indices,
        eigenvalues=retained_eigenvalues,
        eigenvectors=retained_eigenvectors,
        condition_number=condition_number,
        jitter=block_jitter,
    )

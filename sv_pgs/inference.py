from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from scipy.special import expit, logsumexp

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import GraphEdges, TieMap, VariantRecord
from sv_pgs.graph import CorrelationBlock, correlation_blocks
from sv_pgs.operator import GenotypeOperator, matvec, pcg_solve, rmatvec, weighted_column_norms


@dataclass(slots=True)
class BlockPosterior:
    indices: np.ndarray
    covariance_diag: np.ndarray
    low_rank: np.ndarray | None
    covariance_dense: np.ndarray | None


@dataclass(slots=True)
class VariationalFitResult:
    alpha: np.ndarray
    beta_reduced: np.ndarray
    beta_variance: np.ndarray
    responsibilities: np.ndarray
    class_mixture_weights: dict[VariantClass, np.ndarray]
    class_variances: dict[VariantClass, np.ndarray]
    sigma_e2: float
    objective_history: list[float]
    validation_history: list[float]
    block_posteriors: list[BlockPosterior]


def fit_variational_em(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
    tie_map: TieMap,
    graph: GraphEdges,
    config: ModelConfig,
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> VariationalFitResult:
    reduced_x = np.asarray(genotypes[:, tie_map.kept_indices], dtype=np.float32)
    reduced_records = [records[index] for index in tie_map.kept_indices]
    class_to_index = {variant_class: idx for idx, variant_class in enumerate(sorted({record.variant_class for record in reduced_records}, key=lambda value: value.value))}
    index_to_class = {idx: variant_class for variant_class, idx in class_to_index.items()}
    class_index = np.asarray([class_to_index[record.variant_class] for record in reduced_records], dtype=np.int32)
    quality = np.clip(np.asarray([record.quality for record in reduced_records], dtype=np.float32), 1e-4, 1.0)
    base_variances = config.base_component_variances().astype(np.float32)
    class_priors = config.class_prior_weights()

    class_pi = np.vstack([class_priors[index_to_class[idx]] for idx in range(len(class_to_index))]).astype(np.float32)
    class_variances = np.vstack([base_variances.copy() for _ in range(len(class_to_index))]).astype(np.float32)

    responsibilities = np.tile(base_variances[None, :], (reduced_x.shape[1], 1)).astype(np.float32)
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
    beta_mean = np.zeros(reduced_x.shape[1], dtype=np.float32)
    beta_var = np.full(reduced_x.shape[1], base_variances[0], dtype=np.float32)
    alpha = np.zeros(covariates.shape[1], dtype=np.float32)
    sigma_e2 = 1.0

    operator = GenotypeOperator.from_numpy(reduced_x, graph, config)
    blocks = correlation_blocks(graph)
    objective_history: list[float] = []
    validation_history: list[float] = []
    block_posteriors: list[BlockPosterior] = []

    y = np.asarray(targets, dtype=np.float32)
    c = np.asarray(covariates, dtype=np.float32)
    v_x = reduced_x

    for _ in range(config.max_outer_iters):
        eta = np.asarray(matvec(operator, jnp.asarray(beta_mean))) + c @ alpha
        if config.trait_type == TraitType.BINARY:
            weights = _polya_gamma_expectation(eta, config.pg_min_weight)
            kappa = y - 0.5
            alpha = _solve_covariates_binary(c, kappa, weights, eta - c @ alpha)
            rhs_sample = kappa - weights * (c @ alpha)
            sigma_e2 = 1.0
        else:
            residual = y - eta
            sigma_e2 = float(np.mean(residual * residual) + config.sigma_e_prior)
            weights = np.full_like(y, 1.0 / sigma_e2)
            alpha = _solve_covariates_gaussian(c, y, weights, eta - c @ alpha)
            rhs_sample = weights * (y - c @ alpha)

        prior_precision = _effective_prior_precision(responsibilities, class_variances, class_index, quality)
        rhs = np.asarray(rmatvec(operator, jnp.asarray(rhs_sample, dtype=jnp.float32)))
        diag = np.asarray(weighted_column_norms(operator, jnp.asarray(weights, dtype=jnp.float32))) + prior_precision
        diag = diag + _laplacian_diagonal(graph, reduced_x.shape[1])
        beta_mean = np.asarray(
            pcg_solve(
                operator,
                jnp.asarray(rhs, dtype=jnp.float32),
                jnp.asarray(weights, dtype=jnp.float32),
                jnp.asarray(prior_precision, dtype=jnp.float32),
                jnp.asarray(np.maximum(diag, 1e-4), dtype=jnp.float32),
                jnp.asarray(beta_mean, dtype=jnp.float32),
                config.pcg_tolerance,
                config.max_inner_pcg_iters,
            )
        )

        block_posteriors = _refresh_block_posteriors(
            blocks=blocks,
            graph=graph,
            genotypes=v_x,
            sample_weights=weights,
            prior_precision=prior_precision,
            config=config,
        )
        beta_var = np.full(reduced_x.shape[1], base_variances[0], dtype=np.float32)
        for block in block_posteriors:
            beta_var[block.indices] = block.covariance_diag

        responsibilities = _update_responsibilities(
            beta_mean=beta_mean,
            beta_var=beta_var,
            class_index=class_index,
            quality=quality,
            class_pi=class_pi,
            class_variances=class_variances,
        )
        class_pi = _update_class_mixtures(class_index, responsibilities, class_pi, config)
        class_variances = _update_class_variances(
            class_index=class_index,
            responsibilities=responsibilities,
            beta_mean=beta_mean,
            beta_var=beta_var,
            quality=quality,
            class_variances=class_variances,
            config=config,
        )

        objective = _surrogate_objective(
            trait_type=config.trait_type,
            y=y,
            eta=np.asarray(matvec(operator, jnp.asarray(beta_mean))) + c @ alpha,
            weights=weights,
            beta_mean=beta_mean,
            prior_precision=prior_precision,
            graph=graph,
        )
        objective_history.append(float(objective))
        if validation_data is not None:
            val_x, val_c, val_y = validation_data
            validation_history.append(
                float(
                    _validation_metric(
                        trait_type=config.trait_type,
                        genotypes=val_x,
                        covariates=val_c,
                        beta=beta_mean,
                        alpha=alpha,
                        targets=val_y,
                    )
                )
            )
        if len(objective_history) >= 2 and abs(objective_history[-1] - objective_history[-2]) < config.convergence_tolerance:
            break

    return VariationalFitResult(
        alpha=alpha,
        beta_reduced=beta_mean,
        beta_variance=beta_var,
        responsibilities=responsibilities,
        class_mixture_weights={index_to_class[idx]: class_pi[idx] for idx in range(class_pi.shape[0])},
        class_variances={index_to_class[idx]: class_variances[idx] for idx in range(class_variances.shape[0])},
        sigma_e2=sigma_e2,
        objective_history=objective_history,
        validation_history=validation_history,
        block_posteriors=block_posteriors,
    )


def _polya_gamma_expectation(eta: np.ndarray, min_weight: float) -> np.ndarray:
    abs_eta = np.abs(eta)
    small = abs_eta < 1e-4
    safe = np.where(small, 1.0, abs_eta)
    weights = 0.5 * np.tanh(safe / 2.0) / safe
    weights = np.where(small, 0.25, weights)
    return np.maximum(weights.astype(np.float32), min_weight)


def _solve_covariates_binary(covariates: np.ndarray, kappa: np.ndarray, weights: np.ndarray, xb: np.ndarray) -> np.ndarray:
    lhs = covariates.T @ (weights[:, None] * covariates)
    rhs = covariates.T @ (kappa - weights * xb)
    jitter = np.eye(lhs.shape[0], dtype=np.float32) * 1e-6
    return np.linalg.solve(lhs + jitter, rhs).astype(np.float32)


def _solve_covariates_gaussian(covariates: np.ndarray, y: np.ndarray, weights: np.ndarray, xb: np.ndarray) -> np.ndarray:
    lhs = covariates.T @ (weights[:, None] * covariates)
    rhs = covariates.T @ (weights * (y - xb))
    jitter = np.eye(lhs.shape[0], dtype=np.float32) * 1e-6
    return np.linalg.solve(lhs + jitter, rhs).astype(np.float32)


def _effective_prior_precision(
    responsibilities: np.ndarray,
    class_variances: np.ndarray,
    class_index: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    precisions = 1.0 / (quality[:, None] * class_variances[class_index])
    return np.sum(responsibilities * precisions, axis=1).astype(np.float32)


def _update_responsibilities(
    beta_mean: np.ndarray,
    beta_var: np.ndarray,
    class_index: np.ndarray,
    quality: np.ndarray,
    class_pi: np.ndarray,
    class_variances: np.ndarray,
) -> np.ndarray:
    expected_sq = (beta_mean * beta_mean + beta_var)[:, None]
    local_variances = class_variances[class_index]
    logits = (
        np.log(class_pi[class_index] + 1e-12)
        - 0.5 * np.log(local_variances + 1e-12)
        - 0.5 * expected_sq / (quality[:, None] * local_variances + 1e-12)
    )
    logits = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits).astype(np.float32)


def _update_class_mixtures(
    class_index: np.ndarray,
    responsibilities: np.ndarray,
    current_pi: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    updated = np.zeros_like(current_pi)
    for class_id in range(current_pi.shape[0]):
        mask = class_index == class_id
        prior = config.dirichlet_strength * current_pi[class_id]
        updated[class_id] = responsibilities[mask].sum(axis=0) + prior
        updated[class_id] /= updated[class_id].sum()
    return updated.astype(np.float32)


def _update_class_variances(
    class_index: np.ndarray,
    responsibilities: np.ndarray,
    beta_mean: np.ndarray,
    beta_var: np.ndarray,
    quality: np.ndarray,
    class_variances: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    updated = np.zeros_like(class_variances)
    expected_sq = beta_mean * beta_mean + beta_var
    global_log = np.zeros(class_variances.shape[1], dtype=np.float32)
    global_weight = np.zeros(class_variances.shape[1], dtype=np.float32)

    for class_id in range(class_variances.shape[0]):
        mask = class_index == class_id
        weight = responsibilities[mask]
        numerator = (weight * (expected_sq[mask, None] / quality[mask, None])).sum(axis=0) + 1e-6
        denominator = weight.sum(axis=0) + 1e-6
        local = numerator / denominator
        global_log += weight.sum(axis=0) * np.log(local)
        global_weight += weight.sum(axis=0)
        updated[class_id] = local

    global_mean = np.exp(global_log / np.maximum(global_weight, 1e-6))
    for class_id in range(updated.shape[0]):
        local_log = np.log(np.maximum(updated[class_id], 1e-8))
        shrunk = (1.0 - config.variance_shrinkage) * local_log + config.variance_shrinkage * np.log(global_mean)
        updated[class_id] = _enforce_ordered_variances(np.exp(shrunk), config)
    return updated.astype(np.float32)


def _enforce_ordered_variances(variances: np.ndarray, config: ModelConfig) -> np.ndarray:
    log_var = np.log(np.maximum(variances, config.prior_floor_variance))
    ordered = np.empty_like(log_var)
    ordered[0] = max(log_var[0], np.log(config.prior_floor_variance))
    for index in range(1, log_var.shape[0]):
        ordered[index] = max(log_var[index], ordered[index - 1] + config.variance_min_gap_log)
    return np.exp(ordered).astype(np.float32)


def _laplacian_diagonal(graph: GraphEdges, variant_count: int) -> np.ndarray:
    diagonal = np.zeros(variant_count, dtype=np.float32)
    for src, dst, weight in zip(graph.src, graph.dst, graph.weight, strict=True):
        diagonal[src] += weight
        diagonal[dst] += weight
    return diagonal


def _refresh_block_posteriors(
    blocks: Sequence[CorrelationBlock],
    graph: GraphEdges,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    prior_precision: np.ndarray,
    config: ModelConfig,
) -> list[BlockPosterior]:
    posteriors: list[BlockPosterior] = []
    for block in blocks:
        local_hessian = _build_local_hessian(block.indices, graph, genotypes, sample_weights, prior_precision)
        if block.indices.shape[0] <= config.covariance_max_block_exact:
            covariance = np.linalg.inv(local_hessian).astype(np.float32)
            posteriors.append(
                BlockPosterior(
                    indices=block.indices,
                    covariance_diag=np.diag(covariance).astype(np.float32),
                    low_rank=None,
                    covariance_dense=covariance,
                )
            )
            continue

        if block.indices.shape[0] <= config.covariance_max_block_dense:
            covariance = np.linalg.inv(local_hessian).astype(np.float32)
            diag = np.diag(covariance).astype(np.float32)
            centered = covariance - np.diag(diag)
            eigvals, eigvecs = np.linalg.eigh(centered)
            keep = eigvals > 1e-8
            eigvals = eigvals[keep]
            eigvecs = eigvecs[:, keep]
            if eigvals.shape[0] > config.covariance_low_rank:
                eigvals = eigvals[-config.covariance_low_rank :]
                eigvecs = eigvecs[:, -config.covariance_low_rank :]
            low_rank = eigvecs * np.sqrt(eigvals)
            posteriors.append(
                BlockPosterior(
                    indices=block.indices,
                    covariance_diag=diag,
                    low_rank=low_rank.astype(np.float32),
                    covariance_dense=None,
                )
            )
            continue

        diag_precision = np.diag(local_hessian).astype(np.float32)
        posteriors.append(
            BlockPosterior(
                indices=block.indices,
                covariance_diag=(1.0 / np.maximum(diag_precision, 1e-6)).astype(np.float32),
                low_rank=None,
                covariance_dense=None,
            )
        )
    return posteriors


def _build_local_hessian(
    block_indices: np.ndarray,
    graph: GraphEdges,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    prior_precision: np.ndarray,
) -> np.ndarray:
    local_x = genotypes[:, block_indices]
    hessian = local_x.T @ (sample_weights[:, None] * local_x)
    hessian += np.diag(prior_precision[block_indices])
    local_lookup = {global_idx: local_idx for local_idx, global_idx in enumerate(block_indices.tolist())}
    for src, dst, sign, weight in zip(graph.src, graph.dst, graph.sign, graph.weight, strict=True):
        if src not in local_lookup or dst not in local_lookup:
            continue
        left = local_lookup[src]
        right = local_lookup[dst]
        hessian[left, left] += weight
        hessian[right, right] += weight
        hessian[left, right] -= sign * weight
        hessian[right, left] -= sign * weight
    hessian += np.eye(hessian.shape[0], dtype=np.float32) * 1e-6
    return hessian.astype(np.float32)


def _surrogate_objective(
    trait_type: TraitType,
    y: np.ndarray,
    eta: np.ndarray,
    weights: np.ndarray,
    beta_mean: np.ndarray,
    prior_precision: np.ndarray,
    graph: GraphEdges,
) -> float:
    if trait_type == TraitType.BINARY:
        likelihood = float(np.sum(y * np.log(expit(eta) + 1e-8) + (1.0 - y) * np.log(1.0 - expit(eta) + 1e-8)))
    else:
        residual = y - eta
        likelihood = float(-0.5 * np.sum(weights * residual * residual))
    prior = float(-0.5 * np.sum(prior_precision * beta_mean * beta_mean))
    graph_penalty = 0.0
    for src, dst, sign, weight in zip(graph.src, graph.dst, graph.sign, graph.weight, strict=True):
        diff = beta_mean[src] - sign * beta_mean[dst]
        graph_penalty -= 0.5 * float(weight * diff * diff)
    return likelihood + prior + graph_penalty


def _validation_metric(
    trait_type: TraitType,
    genotypes: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    targets: np.ndarray,
) -> float:
    eta = genotypes @ beta + covariates @ alpha
    if trait_type == TraitType.BINARY:
        prob = expit(eta)
        return -float(np.mean(targets * np.log(prob + 1e-8) + (1.0 - targets) * np.log(1.0 - prob + 1e-8)))
    residual = targets - eta
    return float(np.mean(residual * residual))

"""Structured variational EM inference with JAX hot loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

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
    reduced_genotypes = np.asarray(genotypes[:, tie_map.kept_indices], dtype=np.float32)
    reduced_records = [records[int(variant_index)] for variant_index in tie_map.kept_indices]
    class_index_lookup, inverse_class_lookup = _class_mappings(reduced_records)
    class_indices_host = np.asarray(
        [class_index_lookup[variant_record.variant_class] for variant_record in reduced_records],
        dtype=np.int32,
    )
    quality_host = np.clip(
        np.asarray([variant_record.quality for variant_record in reduced_records], dtype=np.float32),
        1e-4,
        1.0,
    )
    base_component_variances = config.base_component_variances().astype(np.float32)
    class_prior_lookup = config.class_prior_weights()

    genotype_operator = GenotypeOperator.from_numpy(reduced_genotypes, graph, config)
    blocks = correlation_blocks(graph)

    target_vector = jnp.asarray(targets, dtype=jnp.float32)
    covariate_matrix = jnp.asarray(covariates, dtype=jnp.float32)
    class_indices = jnp.asarray(class_indices_host, dtype=jnp.int32)
    quality_vector = jnp.asarray(quality_host, dtype=jnp.float32)

    class_count = len(class_index_lookup)
    class_mixture_weights = jnp.asarray(
        np.vstack([class_prior_lookup[inverse_class_lookup[class_index]] for class_index in range(class_count)]),
        dtype=jnp.float32,
    )
    class_variances = jnp.asarray(
        np.tile(base_component_variances, (class_count, 1)),
        dtype=jnp.float32,
    )

    initial_component_weights = np.tile(base_component_variances[None, :], (reduced_genotypes.shape[1], 1))
    responsibilities = jnp.asarray(
        initial_component_weights / initial_component_weights.sum(axis=1, keepdims=True),
        dtype=jnp.float32,
    )
    coefficient_mean = jnp.zeros(reduced_genotypes.shape[1], dtype=jnp.float32)
    coefficient_variance = jnp.full(reduced_genotypes.shape[1], base_component_variances[0], dtype=jnp.float32)
    covariate_coefficients = jnp.zeros(covariate_matrix.shape[1], dtype=jnp.float32)
    residual_variance = 1.0

    graph_diagonal = _laplacian_diagonal(
        source_indices=jnp.asarray(graph.src, dtype=jnp.int32),
        destination_indices=jnp.asarray(graph.dst, dtype=jnp.int32),
        edge_weights=jnp.asarray(graph.weight, dtype=jnp.float32),
        variant_count=reduced_genotypes.shape[1],
    )

    prepared_validation = None
    if validation_data is not None:
        validation_genotypes, validation_covariates, validation_targets = validation_data
        prepared_validation = (
            jnp.asarray(validation_genotypes, dtype=jnp.float32),
            jnp.asarray(validation_covariates, dtype=jnp.float32),
            jnp.asarray(validation_targets, dtype=jnp.float32),
        )

    objective_history: list[float] = []
    validation_history: list[float] = []
    block_posteriors: list[BlockPosterior] = []

    outer_iteration = 0
    while outer_iteration < config.max_outer_iters:
        linear_predictor = matvec(genotype_operator, coefficient_mean) + covariate_matrix @ covariate_coefficients
        sample_weights, response_vector, residual_variance = _likelihood_update(
            trait_type=config.trait_type,
            targets=target_vector,
            linear_predictor=linear_predictor,
            covariate_matrix=covariate_matrix,
            genotype_prediction=matvec(genotype_operator, coefficient_mean),
            sigma_e_prior=config.sigma_e_prior,
            minimum_pg_weight=config.pg_min_weight,
        )
        covariate_coefficients = _solve_covariates(
            covariate_matrix=covariate_matrix,
            response_vector=response_vector,
            sample_weights=sample_weights,
        )

        coefficient_right_hand_side = _coefficient_right_hand_side(
            trait_type=config.trait_type,
            target_vector=target_vector,
            sample_weights=sample_weights,
            covariate_matrix=covariate_matrix,
            covariate_coefficients=covariate_coefficients,
        )
        prior_precision = _effective_prior_precision(
            responsibilities=responsibilities,
            class_variances=class_variances,
            class_indices=class_indices,
            quality_vector=quality_vector,
        )
        projected_right_hand_side = rmatvec(genotype_operator, coefficient_right_hand_side)
        preconditioner_diagonal = (
            weighted_column_norms(genotype_operator, sample_weights)
            + prior_precision
            + graph_diagonal
        )
        coefficient_mean = pcg_solve(
            operator=genotype_operator,
            right_hand_side=projected_right_hand_side,
            sample_weights=sample_weights,
            prior_precision=prior_precision,
            preconditioner_diagonal=jnp.maximum(preconditioner_diagonal, 1e-4),
            initial_coefficients=coefficient_mean,
            tolerance=config.pcg_tolerance,
            maximum_iterations=config.max_inner_pcg_iters,
        )

        block_posteriors = _refresh_block_posteriors(
            blocks=blocks,
            graph=graph,
            genotypes=reduced_genotypes,
            sample_weights=np.asarray(sample_weights),
            prior_precision=np.asarray(prior_precision),
            config=config,
        )
        coefficient_variance_host = np.full(reduced_genotypes.shape[1], base_component_variances[0], dtype=np.float32)
        for block_posterior in block_posteriors:
            coefficient_variance_host[block_posterior.indices] = block_posterior.covariance_diag
        coefficient_variance = jnp.asarray(coefficient_variance_host, dtype=jnp.float32)

        responsibilities = _update_responsibilities(
            coefficient_mean=coefficient_mean,
            coefficient_variance=coefficient_variance,
            class_indices=class_indices,
            quality_vector=quality_vector,
            class_mixture_weights=class_mixture_weights,
            class_variances=class_variances,
        )
        class_mixture_weights = _update_class_mixture_weights(
            class_indices=class_indices,
            responsibilities=responsibilities,
            current_mixture_weights=class_mixture_weights,
            class_count=class_count,
            dirichlet_strength=config.dirichlet_strength,
        )
        class_variances = _update_class_variances(
            class_indices=class_indices,
            responsibilities=responsibilities,
            coefficient_mean=coefficient_mean,
            coefficient_variance=coefficient_variance,
            quality_vector=quality_vector,
            class_count=class_count,
            shrinkage=config.variance_shrinkage,
            floor_variance=config.prior_floor_variance,
            minimum_log_gap=config.variance_min_gap_log,
        )

        updated_linear_predictor = matvec(genotype_operator, coefficient_mean) + covariate_matrix @ covariate_coefficients
        objective_history.append(
            float(
                _surrogate_objective(
                    trait_type=config.trait_type,
                    targets=target_vector,
                    linear_predictor=updated_linear_predictor,
                    sample_weights=sample_weights,
                    coefficient_mean=coefficient_mean,
                    prior_precision=prior_precision,
                    graph=graph,
                )
            )
        )

        if prepared_validation is not None:
            validation_genotypes, validation_covariates, validation_targets = prepared_validation
            validation_linear_predictor = validation_genotypes @ coefficient_mean + validation_covariates @ covariate_coefficients
            validation_history.append(
                float(
                    _validation_metric(
                        trait_type=config.trait_type,
                        targets=validation_targets,
                        linear_predictor=validation_linear_predictor,
                    )
                )
            )

        outer_iteration += 1
        if len(objective_history) >= 2:
            objective_delta = abs(objective_history[-1] - objective_history[-2])
            if objective_delta < config.convergence_tolerance:
                break

    class_mixture_weights_host = np.asarray(class_mixture_weights)
    class_variances_host = np.asarray(class_variances)
    return VariationalFitResult(
        alpha=np.asarray(covariate_coefficients),
        beta_reduced=np.asarray(coefficient_mean),
        beta_variance=np.asarray(coefficient_variance),
        responsibilities=np.asarray(responsibilities),
        class_mixture_weights={
            inverse_class_lookup[class_index]: class_mixture_weights_host[class_index]
            for class_index in range(class_count)
        },
        class_variances={
            inverse_class_lookup[class_index]: class_variances_host[class_index]
            for class_index in range(class_count)
        },
        sigma_e2=float(residual_variance),
        objective_history=objective_history,
        validation_history=validation_history,
        block_posteriors=block_posteriors,
    )


def _class_mappings(
    records: Sequence[VariantRecord],
) -> tuple[dict[VariantClass, int], dict[int, VariantClass]]:
    unique_classes = sorted({record.variant_class for record in records}, key=lambda variant_class: variant_class.value)
    class_index_lookup = {
        variant_class: class_index for class_index, variant_class in enumerate(unique_classes)
    }
    inverse_lookup = {
        class_index: variant_class for variant_class, class_index in class_index_lookup.items()
    }
    return class_index_lookup, inverse_lookup


def _likelihood_update(
    trait_type: TraitType,
    targets: jnp.ndarray,
    linear_predictor: jnp.ndarray,
    covariate_matrix: jnp.ndarray,
    genotype_prediction: jnp.ndarray,
    sigma_e_prior: float,
    minimum_pg_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    if trait_type == TraitType.BINARY:
        sample_weights = _polya_gamma_expectation(linear_predictor, minimum_pg_weight)
        response_vector = targets - 0.5 - sample_weights * genotype_prediction
        return sample_weights, response_vector, 1.0

    residual_vector = targets - linear_predictor
    residual_variance = float(jnp.mean(residual_vector * residual_vector) + sigma_e_prior)
    sample_weights = jnp.full(targets.shape[0], 1.0 / residual_variance, dtype=jnp.float32)
    response_vector = sample_weights * (targets - genotype_prediction)
    return sample_weights, response_vector, residual_variance


def _solve_covariates(
    covariate_matrix: jnp.ndarray,
    response_vector: jnp.ndarray,
    sample_weights: jnp.ndarray,
) -> jnp.ndarray:
    weighted_covariates = jnp.transpose(covariate_matrix) * sample_weights[None, :]
    normal_matrix = weighted_covariates @ covariate_matrix
    right_hand_side = jnp.transpose(covariate_matrix) @ response_vector
    ridge_jitter = 1e-6 * jnp.eye(covariate_matrix.shape[1], dtype=jnp.float32)
    return jnp.linalg.solve(normal_matrix + ridge_jitter, right_hand_side)


def _coefficient_right_hand_side(
    trait_type: TraitType,
    target_vector: jnp.ndarray,
    sample_weights: jnp.ndarray,
    covariate_matrix: jnp.ndarray,
    covariate_coefficients: jnp.ndarray,
) -> jnp.ndarray:
    if trait_type == TraitType.BINARY:
        return target_vector - 0.5 - sample_weights * (covariate_matrix @ covariate_coefficients)
    return sample_weights * (target_vector - covariate_matrix @ covariate_coefficients)


def _polya_gamma_expectation(
    linear_predictor: jnp.ndarray,
    minimum_weight: float,
) -> jnp.ndarray:
    absolute_linear_predictor = jnp.abs(linear_predictor)
    small_mask = absolute_linear_predictor < 1e-4
    safe_linear_predictor = jnp.where(small_mask, 1.0, absolute_linear_predictor)
    weights = 0.5 * jnp.tanh(safe_linear_predictor / 2.0) / safe_linear_predictor
    weights = jnp.where(small_mask, 0.25, weights)
    return jnp.maximum(weights, minimum_weight)


def _effective_prior_precision(
    responsibilities: jnp.ndarray,
    class_variances: jnp.ndarray,
    class_indices: jnp.ndarray,
    quality_vector: jnp.ndarray,
) -> jnp.ndarray:
    local_variances = class_variances[class_indices]
    component_precisions = 1.0 / (quality_vector[:, None] * local_variances + 1e-30)
    return jnp.sum(responsibilities * component_precisions, axis=1)


def _update_responsibilities(
    coefficient_mean: jnp.ndarray,
    coefficient_variance: jnp.ndarray,
    class_indices: jnp.ndarray,
    quality_vector: jnp.ndarray,
    class_mixture_weights: jnp.ndarray,
    class_variances: jnp.ndarray,
) -> jnp.ndarray:
    expected_squared_effect = (coefficient_mean * coefficient_mean + coefficient_variance)[:, None]
    local_variances = class_variances[class_indices]
    local_mixture_weights = class_mixture_weights[class_indices]
    unnormalized_log_weights = (
        jnp.log(local_mixture_weights + 1e-12)
        - 0.5 * jnp.log(local_variances + 1e-12)
        - 0.5 * expected_squared_effect / (quality_vector[:, None] * local_variances + 1e-12)
    )
    normalized_log_weights = unnormalized_log_weights - jax.nn.logsumexp(
        unnormalized_log_weights,
        axis=1,
        keepdims=True,
    )
    return jnp.exp(normalized_log_weights)


def _update_class_mixture_weights(
    class_indices: jnp.ndarray,
    responsibilities: jnp.ndarray,
    current_mixture_weights: jnp.ndarray,
    class_count: int,
    dirichlet_strength: float,
) -> jnp.ndarray:
    one_hot_classes = jax.nn.one_hot(class_indices, class_count)
    responsibility_sums = jnp.transpose(one_hot_classes) @ responsibilities
    posterior_weights = responsibility_sums + dirichlet_strength * current_mixture_weights
    return posterior_weights / posterior_weights.sum(axis=1, keepdims=True)


def _update_class_variances(
    class_indices: jnp.ndarray,
    responsibilities: jnp.ndarray,
    coefficient_mean: jnp.ndarray,
    coefficient_variance: jnp.ndarray,
    quality_vector: jnp.ndarray,
    class_count: int,
    shrinkage: float,
    floor_variance: float,
    minimum_log_gap: float,
) -> jnp.ndarray:
    expected_squared_effect = coefficient_mean * coefficient_mean + coefficient_variance
    weighted_squared_effect = responsibilities * (expected_squared_effect / quality_vector)[:, None]
    one_hot_classes = jax.nn.one_hot(class_indices, class_count)
    per_class_expected_effect = jnp.transpose(one_hot_classes) @ weighted_squared_effect
    per_class_responsibility = jnp.transpose(one_hot_classes) @ responsibilities
    local_variances = per_class_expected_effect / jnp.maximum(per_class_responsibility, 1e-6)
    global_log_variances = (
        jnp.sum(per_class_responsibility * jnp.log(jnp.maximum(local_variances, 1e-20)), axis=0)
        / jnp.maximum(per_class_responsibility.sum(axis=0), 1e-6)
    )
    global_variances = jnp.exp(global_log_variances)
    shrunk_log_variances = (
        (1.0 - shrinkage) * jnp.log(jnp.maximum(local_variances, 1e-20))
        + shrinkage * jnp.log(global_variances + 1e-20)[None, :]
    )
    return jax.vmap(
        lambda variance_row: _enforce_ordered_variances(
            jnp.exp(shrunk_log_variances[variance_row]),
            floor_variance,
            minimum_log_gap,
        )
    )(jnp.arange(local_variances.shape[0]))


def _enforce_ordered_variances(
    variance_row: jnp.ndarray,
    floor_variance: float,
    minimum_log_gap: float,
) -> jnp.ndarray:
    bounded_row = jnp.maximum(variance_row, floor_variance)
    initial_log_variance = jnp.log(bounded_row[0])

    def scan_step(previous_log_variance: jnp.ndarray, current_variance: jnp.ndarray):
        current_log_variance = jnp.log(jnp.maximum(current_variance, floor_variance))
        updated_log_variance = jnp.maximum(current_log_variance, previous_log_variance + minimum_log_gap)
        return updated_log_variance, updated_log_variance

    remaining_log_variances = lax.scan(scan_step, initial_log_variance, bounded_row[1:])[1]
    return jnp.exp(jnp.concatenate([initial_log_variance[None], remaining_log_variances]))


def _laplacian_diagonal(
    source_indices: jnp.ndarray,
    destination_indices: jnp.ndarray,
    edge_weights: jnp.ndarray,
    variant_count: int,
) -> jnp.ndarray:
    diagonal = jnp.zeros(variant_count, dtype=jnp.float32)
    diagonal = diagonal.at[source_indices].add(edge_weights)
    diagonal = diagonal.at[destination_indices].add(edge_weights)
    return diagonal


def _binary_log_likelihood(
    targets: jnp.ndarray,
    linear_predictor: jnp.ndarray,
) -> jnp.ndarray:
    probabilities = jax.nn.sigmoid(linear_predictor)
    return (
        targets * jnp.log(probabilities + 1e-8)
        + (1.0 - targets) * jnp.log(1.0 - probabilities + 1e-8)
    )


def _surrogate_objective(
    trait_type: TraitType,
    targets: jnp.ndarray,
    linear_predictor: jnp.ndarray,
    sample_weights: jnp.ndarray,
    coefficient_mean: jnp.ndarray,
    prior_precision: jnp.ndarray,
    graph: GraphEdges,
) -> jnp.ndarray:
    if trait_type == TraitType.BINARY:
        likelihood = jnp.sum(_binary_log_likelihood(targets, linear_predictor))
    else:
        residual_vector = targets - linear_predictor
        likelihood = -0.5 * jnp.sum(sample_weights * residual_vector * residual_vector)

    prior_term = -0.5 * jnp.sum(prior_precision * coefficient_mean * coefficient_mean)
    if graph.src.shape[0] == 0:
        graph_term = 0.0
    else:
        coefficient_difference = coefficient_mean[graph.src] - graph.sign * coefficient_mean[graph.dst]
        graph_term = -0.5 * jnp.sum(jnp.asarray(graph.weight) * coefficient_difference * coefficient_difference)
    return likelihood + prior_term + graph_term


def _validation_metric(
    trait_type: TraitType,
    targets: jnp.ndarray,
    linear_predictor: jnp.ndarray,
) -> jnp.ndarray:
    if trait_type == TraitType.BINARY:
        return -jnp.mean(_binary_log_likelihood(targets, linear_predictor))
    residual_vector = targets - linear_predictor
    return jnp.mean(residual_vector * residual_vector)


def _refresh_block_posteriors(
    blocks: Sequence[CorrelationBlock],
    graph: GraphEdges,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    prior_precision: np.ndarray,
    config: ModelConfig,
) -> list[BlockPosterior]:
    block_posteriors: list[BlockPosterior] = []
    for correlation_block in blocks:
        local_hessian = _build_local_hessian(
            block_indices=correlation_block.indices,
            graph=graph,
            genotypes=genotypes,
            sample_weights=sample_weights,
            prior_precision=prior_precision,
        )
        block_size = correlation_block.indices.shape[0]
        if block_size <= config.covariance_max_block_exact:
            covariance_matrix = np.linalg.inv(local_hessian).astype(np.float32)
            block_posteriors.append(
                BlockPosterior(
                    indices=correlation_block.indices,
                    covariance_diag=np.diag(covariance_matrix).astype(np.float32),
                    low_rank=None,
                    covariance_dense=covariance_matrix,
                )
            )
            continue

        if block_size <= config.covariance_max_block_dense:
            covariance_matrix = np.linalg.inv(local_hessian).astype(np.float32)
            covariance_diagonal = np.diag(covariance_matrix).astype(np.float32)
            centered_covariance = covariance_matrix - np.diag(covariance_diagonal)
            eigenvalues, eigenvectors = np.linalg.eigh(centered_covariance)
            positive_mask = eigenvalues > 1e-8
            eigenvalues = eigenvalues[positive_mask]
            eigenvectors = eigenvectors[:, positive_mask]
            if eigenvalues.shape[0] > config.covariance_low_rank:
                eigenvalues = eigenvalues[-config.covariance_low_rank :]
                eigenvectors = eigenvectors[:, -config.covariance_low_rank :]
            block_posteriors.append(
                BlockPosterior(
                    indices=correlation_block.indices,
                    covariance_diag=covariance_diagonal,
                    low_rank=(eigenvectors * np.sqrt(eigenvalues)).astype(np.float32),
                    covariance_dense=None,
                )
            )
            continue

        local_precision_diagonal = np.diag(local_hessian).astype(np.float32)
        block_posteriors.append(
            BlockPosterior(
                indices=correlation_block.indices,
                covariance_diag=(1.0 / np.maximum(local_precision_diagonal, 1e-6)).astype(np.float32),
                low_rank=None,
                covariance_dense=None,
            )
        )
    return block_posteriors


def _build_local_hessian(
    block_indices: np.ndarray,
    graph: GraphEdges,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    prior_precision: np.ndarray,
) -> np.ndarray:
    local_genotypes = genotypes[:, block_indices]
    local_hessian = np.transpose(local_genotypes) @ (sample_weights[:, None] * local_genotypes)
    local_hessian += np.diag(prior_precision[block_indices])
    local_index_lookup = {
        int(global_variant_index): local_variant_index
        for local_variant_index, global_variant_index in enumerate(block_indices.tolist())
    }
    for source_index, destination_index, edge_sign, edge_weight in zip(
        graph.src,
        graph.dst,
        graph.sign,
        graph.weight,
        strict=True,
    ):
        if source_index not in local_index_lookup or destination_index not in local_index_lookup:
            continue
        local_source = local_index_lookup[int(source_index)]
        local_destination = local_index_lookup[int(destination_index)]
        local_hessian[local_source, local_source] += edge_weight
        local_hessian[local_destination, local_destination] += edge_weight
        local_hessian[local_source, local_destination] -= edge_sign * edge_weight
        local_hessian[local_destination, local_source] -= edge_sign * edge_weight
    local_hessian += np.eye(local_hessian.shape[0], dtype=np.float32) * 1e-6
    return local_hessian.astype(np.float32)

"""Matrix-free genotype operator for tiled JAX matvecs and PCG solves."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from sv_pgs.config import ModelConfig
from sv_pgs.data import GraphEdges


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GenotypeOperator:
    genotype_tiles: jnp.ndarray
    tile_mask: jnp.ndarray
    graph_source_indices: jnp.ndarray
    graph_destination_indices: jnp.ndarray
    graph_edge_signs: jnp.ndarray
    graph_edge_weights: jnp.ndarray
    variant_count: int
    tile_size: int

    @classmethod
    def from_numpy(
        cls,
        genotypes: np.ndarray,
        graph: GraphEdges,
        config: ModelConfig,
    ) -> "GenotypeOperator":
        variant_major_genotypes = np.asarray(np.transpose(genotypes), dtype=np.float32)
        variant_count, sample_count = variant_major_genotypes.shape
        tile_size = config.tile_size
        tile_count = int(np.ceil(variant_count / tile_size))
        padded_variant_count = tile_count * tile_size

        padded_genotypes = np.zeros((padded_variant_count, sample_count), dtype=np.float32)
        padded_genotypes[:variant_count] = variant_major_genotypes

        tile_mask = np.zeros((tile_count, tile_size), dtype=np.float32)
        tile_mask.reshape(-1)[:variant_count] = 1.0

        return cls(
            genotype_tiles=jnp.asarray(padded_genotypes.reshape(tile_count, tile_size, sample_count)),
            tile_mask=jnp.asarray(tile_mask),
            graph_source_indices=jnp.asarray(graph.src, dtype=jnp.int32),
            graph_destination_indices=jnp.asarray(graph.dst, dtype=jnp.int32),
            graph_edge_signs=jnp.asarray(graph.sign, dtype=jnp.float32),
            graph_edge_weights=jnp.asarray(graph.weight, dtype=jnp.float32),
            variant_count=variant_count,
            tile_size=tile_size,
        )

    def pad_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        padded_vector = jnp.zeros(self.genotype_tiles.shape[0] * self.tile_size, dtype=vector.dtype)
        return padded_vector.at[: self.variant_count].set(vector)

    def truncate_vector(self, vector: jnp.ndarray) -> jnp.ndarray:
        return vector[: self.variant_count]

    def tree_flatten(self):
        children = (
            self.genotype_tiles,
            self.tile_mask,
            self.graph_source_indices,
            self.graph_destination_indices,
            self.graph_edge_signs,
            self.graph_edge_weights,
        )
        auxiliary_data = (self.variant_count, self.tile_size)
        return children, auxiliary_data

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        variant_count, tile_size = auxiliary_data
        return cls(
            genotype_tiles=children[0],
            tile_mask=children[1],
            graph_source_indices=children[2],
            graph_destination_indices=children[3],
            graph_edge_signs=children[4],
            graph_edge_weights=children[5],
            variant_count=variant_count,
            tile_size=tile_size,
        )


def _scan_matvec(
    genotype_tiles: jnp.ndarray,
    coefficient_tiles: jnp.ndarray,
    tile_mask: jnp.ndarray,
) -> jnp.ndarray:
    def scan_body(
        accumulated_prediction: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, None]:
        genotype_tile, coefficient_tile, mask_tile = inputs
        updated_prediction = accumulated_prediction + jnp.sum(
            genotype_tile * (coefficient_tile * mask_tile)[:, None],
            axis=0,
        )
        return updated_prediction, None

    initial_prediction = jnp.zeros(genotype_tiles.shape[-1], dtype=jnp.float32)
    final_prediction = lax.scan(scan_body, initial_prediction, (genotype_tiles, coefficient_tiles, tile_mask))[0]
    return final_prediction


def _scan_rmatvec(
    genotype_tiles: jnp.ndarray,
    residual_vector: jnp.ndarray,
    tile_mask: jnp.ndarray,
) -> jnp.ndarray:
    def scan_body(
        _unused_carry: None,
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[None, jnp.ndarray]:
        genotype_tile, mask_tile = inputs
        local_projection = jnp.sum(genotype_tile * residual_vector[None, :], axis=1)
        return None, local_projection * mask_tile

    tile_values = lax.scan(scan_body, None, (genotype_tiles, tile_mask))[1]
    return tile_values.reshape(-1)


def _graph_laplacian_matvec(
    coefficient_vector: jnp.ndarray,
    source_indices: jnp.ndarray,
    destination_indices: jnp.ndarray,
    edge_signs: jnp.ndarray,
    edge_weights: jnp.ndarray,
) -> jnp.ndarray:
    if source_indices.shape[0] == 0:
        return jnp.zeros_like(coefficient_vector)

    laplacian_product = jnp.zeros_like(coefficient_vector)
    source_updates = edge_weights * (
        coefficient_vector[source_indices] - edge_signs * coefficient_vector[destination_indices]
    )
    destination_updates = edge_weights * (
        coefficient_vector[destination_indices] - edge_signs * coefficient_vector[source_indices]
    )
    laplacian_product = laplacian_product.at[source_indices].add(source_updates)
    laplacian_product = laplacian_product.at[destination_indices].add(destination_updates)
    return laplacian_product


@jax.jit
def matvec(operator: GenotypeOperator, coefficients: jnp.ndarray) -> jnp.ndarray:
    padded_coefficients = operator.pad_vector(coefficients)
    coefficient_tiles = padded_coefficients.reshape(operator.genotype_tiles.shape[0], operator.tile_size)
    return _scan_matvec(operator.genotype_tiles, coefficient_tiles, operator.tile_mask)


@jax.jit
def rmatvec(operator: GenotypeOperator, residual_vector: jnp.ndarray) -> jnp.ndarray:
    padded_projection = _scan_rmatvec(operator.genotype_tiles, residual_vector, operator.tile_mask)
    return operator.truncate_vector(padded_projection)


@jax.jit
def weighted_column_norms(
    operator: GenotypeOperator,
    sample_weights: jnp.ndarray,
) -> jnp.ndarray:
    def scan_body(
        _unused_carry: None,
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[None, jnp.ndarray]:
        genotype_tile, mask_tile = inputs
        local_norms = jnp.sum(genotype_tile * genotype_tile * sample_weights[None, :], axis=1)
        return None, local_norms * mask_tile

    tile_values = lax.scan(scan_body, None, (operator.genotype_tiles, operator.tile_mask))[1]
    return operator.truncate_vector(tile_values.reshape(-1))


@jax.jit
def apply_hessian(
    operator: GenotypeOperator,
    coefficients: jnp.ndarray,
    sample_weights: jnp.ndarray,
    prior_precision: jnp.ndarray,
) -> jnp.ndarray:
    projected_prediction = matvec(operator, coefficients)
    weighted_projection = rmatvec(operator, sample_weights * projected_prediction)
    graph_projection = _graph_laplacian_matvec(
        coefficients,
        operator.graph_source_indices,
        operator.graph_destination_indices,
        operator.graph_edge_signs,
        operator.graph_edge_weights,
    )
    return weighted_projection + prior_precision * coefficients + graph_projection


@jax.jit
def pcg_solve(
    operator: GenotypeOperator,
    right_hand_side: jnp.ndarray,
    sample_weights: jnp.ndarray,
    prior_precision: jnp.ndarray,
    preconditioner_diagonal: jnp.ndarray,
    initial_coefficients: jnp.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> jnp.ndarray:
    inverse_preconditioner = 1.0 / jnp.maximum(preconditioner_diagonal, 1e-10)

    initial_residual = right_hand_side - apply_hessian(
        operator,
        initial_coefficients,
        sample_weights,
        prior_precision,
    )
    initial_search_direction = inverse_preconditioner * initial_residual
    initial_residual_dot = jnp.dot(initial_residual, initial_search_direction)
    right_hand_side_norm = jnp.linalg.norm(right_hand_side) + 1e-30

    initial_state = (
        initial_coefficients,
        initial_residual,
        initial_search_direction,
        initial_residual_dot,
        jnp.int32(0),
    )

    def continue_iterations(state):
        normalized_residual = jnp.linalg.norm(state[1]) / right_hand_side_norm
        return (normalized_residual > tolerance) & (state[4] < maximum_iterations)

    def update_state(state):
        coefficient_vector, residual_vector, search_direction, residual_dot, iteration_index = state
        hessian_search_direction = apply_hessian(
            operator,
            search_direction,
            sample_weights,
            prior_precision,
        )
        curvature = jnp.dot(search_direction, hessian_search_direction)
        step_size = residual_dot / jnp.maximum(curvature, 1e-30)
        updated_coefficients = coefficient_vector + step_size * search_direction
        updated_residual = residual_vector - step_size * hessian_search_direction
        preconditioned_residual = inverse_preconditioner * updated_residual
        updated_residual_dot = jnp.dot(updated_residual, preconditioned_residual)
        conjugate_scale = updated_residual_dot / jnp.maximum(residual_dot, 1e-30)
        updated_search_direction = preconditioned_residual + conjugate_scale * search_direction
        return (
            updated_coefficients,
            updated_residual,
            updated_search_direction,
            updated_residual_dot,
            iteration_index + 1,
        )

    final_state = lax.while_loop(continue_iterations, update_state, initial_state)
    return final_state[0]
